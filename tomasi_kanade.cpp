
/* tomasi_kanade.cpp */

#include "tomasi_kanade.hpp"

#include <cassert>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/EigenValues>
#include <Eigen/SVD>

/* Perform Tomasi-Kanade method */
bool TomasiKanade::Run(const PointTracker& pointTracker)
{
    this->mPointTracker = pointTracker;

    const std::size_t numOfFrames = pointTracker.NumOfFrames();
    const std::size_t numOfPoints = pointTracker.NumOfPoints();
    const auto& pointTracks = pointTracker.Tracks();

    /* Create observation matrix */
    this->mObservationMat = Eigen::MatrixXd(2 * numOfFrames, numOfPoints);

    for (std::size_t i = 0; i < numOfFrames; ++i) {
        for (std::size_t j = 0; j < numOfPoints; ++j)
            this->mObservationMat.block<2, 1>(2 * i, j) =
                Eigen::Vector2d(pointTracks[j][i].x, pointTracks[j][i].y);

        /* Compute the centroid for each frame */
        const Eigen::Vector2d centroidPoint {
            this->mObservationMat.row(2 * i).mean(),
            this->mObservationMat.row(2 * i + 1).mean() };

        /* Subtract the centroid for each point */
        for (std::size_t j = 0; j < numOfPoints; ++j)
            this->mObservationMat.block<2, 1>(2 * i, j) -= centroidPoint;

        /* Append to the list of the centroids */
        this->mCentroidPoints.push_back(centroidPoint);
    }

    assert(this->mCentroidPoints.size() == numOfFrames);

    /* Compute singular value decomposition of the observation matrix */
    Eigen::JacobiSVD<Eigen::MatrixXd> jacobiSvd {
        this->mObservationMat, Eigen::ComputeThinU | Eigen::ComputeThinV };

    /* Get the column vectors corresponding to the largest singular values */
    this->mUMat = jacobiSvd.matrixU().leftCols(3);
    this->mVMat = jacobiSvd.matrixV().leftCols(3);
    this->mSingularValuesVec = jacobiSvd.singularValues().head(3);

    /* Compute the motion matrix */
    this->mMotionMat = this->mUMat;
    /* Compute the shape matrix */
    this->mShapeMat = this->mSingularValuesVec.asDiagonal() *
                      this->mVMat.transpose();

    /* Check the size of the matrices */
    assert(this->mMotionMat.rows() == 2 * numOfFrames);
    assert(this->mMotionMat.cols() == 3);
    assert(this->mShapeMat.rows() == 3);
    assert(this->mShapeMat.cols() == numOfPoints);

    return true;
}

/* Perform Euclidean upgrading (orthographic) */
bool TomasiKanade::CorrectOrthographic()
{
    auto coeff = [this](int i, int j, int k, int l) {
        return this->ComputeCoefficientOrthographic(i, j, k, l); };

    /* Construct 6x6 matrix B */
    const Eigen::MatrixXd matB = this->ComputeCoefficientMatrix(coeff);

    Eigen::VectorXd vecB { 6 };
    vecB << 1.0, 1.0, 1.0, 0.0, 0.0, 0.0;

    /* Solve for vector tau */
    const Eigen::VectorXd vecTau = matB.fullPivLu().solve(vecB);

    /* Create metric matrix */
    const double sqrt2 = std::sqrt(2.0);
    this->mMetricMat << vecTau(0), vecTau(5) / sqrt2, vecTau(4) / sqrt2,
                        vecTau(5) / sqrt2, vecTau(1), vecTau(3) / sqrt2,
                        vecTau(4) / sqrt2, vecTau(3) / sqrt2, vecTau(2);

    /* Compute matrix to correct motion matrix and shape matrix */
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver {
        this->mMetricMat };
    const Eigen::Matrix3d matAffineCorrection = eigenSolver.operatorSqrt();

    /* Correct motion matrix */
    this->mMotionMat = this->mUMat * matAffineCorrection;
    /* Correct shape matrix */
    this->mShapeMat = matAffineCorrection.inverse() *
                      this->mSingularValuesVec.asDiagonal() *
                      this->mVMat.transpose();

    /* Check the size of the matrices */
    assert(this->mMotionMat.rows() == 2 * this->mPointTracker.NumOfFrames());
    assert(this->mMotionMat.cols() == 3);
    assert(this->mShapeMat.rows() == 3);
    assert(this->mShapeMat.cols() == this->mPointTracker.NumOfPoints());

    return true;
}

/* Perform Euclidean upgrading (weak perspective) */
bool TomasiKanade::CorrectWeakPerspective()
{
    auto coeff = [this](int i, int j, int k, int l) {
        return this->ComputeCoefficientWeakPerspective(i, j, k, l); };

    /* Construct 6x6 matrix B */
    const Eigen::MatrixXd matB = this->ComputeCoefficientMatrix(coeff);
    /* Perform eigen decomposition */
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver { matB };
    /* Get the eigenvector associated with the smallest eigenvalue */
    const Eigen::VectorXd vecTau = eigenSolver.eigenvectors().col(0);

    /* Create metric matrix */
    const double sqrt2 = std::sqrt(2.0);
    this->mMetricMat << vecTau(0), vecTau(5) / sqrt2, vecTau(4) / sqrt2,
                        vecTau(5) / sqrt2, vecTau(1), vecTau(3) / sqrt2,
                        vecTau(4) / sqrt2, vecTau(3) / sqrt2, vecTau(2);

    /* Negate the metric matrix if the determinant is negative */
    if (this->mMetricMat.determinant() < 0.0)
        this->mMetricMat = -this->mMetricMat;

    /* Compute matrix to correct motion matrix and shape matrix */
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> affineSolver {
        this->mMetricMat };
    const Eigen::Matrix3d matAffineCorrection = affineSolver.operatorSqrt();

    /* Correct motion matrix */
    this->mMotionMat = this->mUMat * matAffineCorrection;
    /* Correct shape matrix */
    this->mShapeMat = matAffineCorrection.inverse() *
                      this->mSingularValuesVec.asDiagonal() *
                      this->mVMat.transpose();

    /* Check the size of the matrices */
    assert(this->mMotionMat.rows() == 2 * this->mPointTracker.NumOfFrames());
    assert(this->mMotionMat.cols() == 3);
    assert(this->mShapeMat.rows() == 3);
    assert(this->mShapeMat.cols() == this->mPointTracker.NumOfPoints());

    return true;
}

/* Estimate camera poses (translational vectors and rotation matrices) */
bool TomasiKanade::EstimateCameraPoseOrthographic()
{
    this->mTranslationVectors.clear();
    this->mTranslationVectors.reserve(this->mPointTracker.NumOfFrames());
    this->mRotationMatrices.clear();
    this->mRotationMatrices.reserve(this->mPointTracker.NumOfFrames());

    for (std::size_t i = 0; i < this->mPointTracker.NumOfFrames(); ++i) {
        Eigen::Vector3d transVec;
        transVec[0] = this->mCentroidPoints[i][0];
        transVec[1] = this->mCentroidPoints[i][1];
        transVec[2] = 0.0;
        
        Eigen::Matrix3d rotationMat;
        rotationMat.col(0) = this->mMotionMat.row(2 * i).transpose();
        rotationMat.col(1) = this->mMotionMat.row(2 * i + 1).transpose();
        rotationMat.col(2) = Eigen::Vector3d::Zero().transpose();

        Eigen::JacobiSVD<Eigen::MatrixXd> jacobiSvd {
            rotationMat, Eigen::ComputeThinU | Eigen::ComputeThinV };

        Eigen::Vector3d diagVec { 1.0, 1.0,
            (jacobiSvd.matrixU() * jacobiSvd.matrixV()
                .transpose()).determinant() };
        Eigen::Matrix3d actualRotationMat =
            jacobiSvd.matrixU() * diagVec.asDiagonal() *
            jacobiSvd.matrixV().transpose();

        transVec = -actualRotationMat * transVec;

        this->mTranslationVectors.push_back(std::move(transVec));
        this->mRotationMatrices.push_back(std::move(actualRotationMat));
    }

    for (std::size_t i = 0; i < this->mPointTracker.NumOfFrames(); ++i) {
        this->mMotionMat.row(2 * i) =
            this->mRotationMatrices[i].col(0).transpose();
        this->mMotionMat.row(2 * i + 1) =
            this->mRotationMatrices[i].col(1).transpose();
    }

    this->mShapeMat =
        (this->mMotionMat.transpose() * this->mMotionMat).inverse() *
        this->mMotionMat.transpose() * this->mObservationMat;

    return true;
}

/* Estimate camera poses (translational vectors and rotation matrices) */
bool TomasiKanade::EstimateCameraPoseWeakPerspective()
{
    this->mTranslationVectors.clear();
    this->mTranslationVectors.reserve(this->mPointTracker.NumOfFrames());
    this->mRotationMatrices.clear();
    this->mRotationMatrices.reserve(this->mPointTracker.NumOfFrames());

    for (std::size_t i = 0; i < this->mPointTracker.NumOfFrames(); ++i) {
        const double metricCond0 = this->mUMat.row(2 * i) *
                                   this->mMetricMat *
                                   this->mUMat.row(2 * i).transpose();
        const double metricCond1 = this->mUMat.row(2 * i + 1) *
                                   this->mMetricMat *
                                   this->mUMat.row(2 * i + 1).transpose();
        Eigen::Vector3d transVec;
        transVec[2] = std::sqrt(2.0 / (metricCond0 + metricCond1));
        transVec[0] = transVec[2] * this->mCentroidPoints[i][0];
        transVec[1] = transVec[2] * this->mCentroidPoints[i][1];
        
        Eigen::Matrix3d rotationMat;
        rotationMat.col(0) = this->mMotionMat.row(2 * i);
        rotationMat.col(1) = this->mMotionMat.row(2 * i + 1);
        rotationMat.col(2) = Eigen::Vector3d::Zero();
        rotationMat *= transVec[2];

        Eigen::JacobiSVD<Eigen::MatrixXd> jacobiSvd {
            rotationMat, Eigen::ComputeThinU | Eigen::ComputeThinV };

        Eigen::Vector3d diagVec { 1.0, 1.0,
            (jacobiSvd.matrixU() * jacobiSvd.matrixV()
                .transpose()).determinant() };
        Eigen::Matrix3d actualRotationMat =
            jacobiSvd.matrixU() * diagVec.asDiagonal() *
            jacobiSvd.matrixV().transpose();

        this->mTranslationVectors.push_back(std::move(transVec));
        this->mRotationMatrices.push_back(std::move(actualRotationMat));
    }

    return true;
}

/* Compute symmetric coefficient matrix */
Eigen::MatrixXd TomasiKanade::ComputeCoefficientMatrix(
    std::function<double(int, int, int, int)> coeff)
{
    const double sqrt2 = std::sqrt(2.0);

    Eigen::MatrixXd matB { 6, 6 };
    matB.row(0) << coeff(0, 0, 0, 0), coeff(0, 0, 1, 1), coeff(0, 0, 2, 2),
                   sqrt2 * coeff(0, 0, 1, 2), sqrt2 * coeff(0, 0, 2, 0),
                   sqrt2 * coeff(0, 0, 0, 1);
    matB.row(1) << coeff(1, 1, 0, 0), coeff(1, 1, 1, 1), coeff(1, 1, 2, 2),
                   sqrt2 * coeff(1, 1, 1, 2), sqrt2 * coeff(1, 1, 2, 0),
                   sqrt2 * coeff(1, 1, 0, 1);
    matB.row(2) << coeff(2, 2, 0, 0), coeff(2, 2, 1, 1), coeff(2, 2, 2, 2),
                   sqrt2 * coeff(2, 2, 1, 2), sqrt2 * coeff(2, 2, 2, 0),
                   sqrt2 * coeff(2, 2, 0, 1);

    matB.row(3) << sqrt2 * coeff(1, 2, 0, 0), sqrt2 * coeff(1, 2, 1, 1),
                   sqrt2 * coeff(1, 2, 2, 2),
                   2.0 * coeff(1, 2, 1, 2), 2.0 * coeff(1, 2, 2, 0),
                   2.0 * coeff(1, 2, 0, 1);
    matB.row(4) << sqrt2 * coeff(2, 0, 0, 0), sqrt2 * coeff(2, 0, 1, 1),
                   sqrt2 * coeff(2, 0, 2, 2),
                   2.0 * coeff(2, 0, 1, 2), 2.0 * coeff(2, 0, 2, 0),
                   2.0 * coeff(2, 0, 0, 1);
    matB.row(5) << sqrt2 * coeff(0, 1, 0, 0), sqrt2 * coeff(0, 1, 1, 1),
                   sqrt2 * coeff(0, 1, 2, 2),
                   2.0 * coeff(0, 1, 1, 2), 2.0 * coeff(0, 1, 2, 0),
                   2.0 * coeff(0, 1, 0, 1);

    return matB;
}

/* Compute coefficient for Euclidean upgrading (orthographic) */
double TomasiKanade::ComputeCoefficientOrthographic(
    int i, int j, int k, int l)
{
    double coeffVal = 0.0;

    for (std::size_t m = 0; m < this->mPointTracker.NumOfFrames(); ++m) {
        const double i1 = this->mUMat(2 * m, i);
        const double j1 = this->mUMat(2 * m, j);
        const double k1 = this->mUMat(2 * m, k);
        const double l1 = this->mUMat(2 * m, l);
        const double i2 = this->mUMat(2 * m + 1, i);
        const double j2 = this->mUMat(2 * m + 1, j);
        const double k2 = this->mUMat(2 * m + 1, k);
        const double l2 = this->mUMat(2 * m + 1, l);

        const double val0 = i1 * j1 * k1 * l1;
        const double val1 = i2 * j2 * k2 * l2;
        const double val2 = i1 * j2 + i2 * j1;
        const double val3 = k1 * l2 + k2 * l1;

        coeffVal += val0 + val1 + (val2 * val3) / 4.0;
    }

    return coeffVal;
}

/* Compute coefficient for Euclidean upgrading (weak perspective) */
double TomasiKanade::ComputeCoefficientWeakPerspective(
    int i, int j, int k, int l)
{
    double coeffVal = 0.0;

    for (std::size_t m = 0; m < this->mPointTracker.NumOfFrames(); ++m) {
        const double i1 = this->mUMat(2 * m, i);
        const double j1 = this->mUMat(2 * m, j);
        const double k1 = this->mUMat(2 * m, k);
        const double l1 = this->mUMat(2 * m, l);
        const double i2 = this->mUMat(2 * m + 1, i);
        const double j2 = this->mUMat(2 * m + 1, j);
        const double k2 = this->mUMat(2 * m + 1, k);
        const double l2 = this->mUMat(2 * m + 1, l);

        const double val0 = i1 * j1 * k1 * l1;
        const double val1 = i1 * j1 * k2 * l2;
        const double val2 = i2 * j2 * k1 * l1;
        const double val3 = i2 * j2 * k2 * l2;

        const double val4 = i1 * j2 * k1 * l2;
        const double val5 = i2 * j1 * k1 * l2;
        const double val6 = i1 * j2 * k2 * l1;
        const double val7 = i2 * j1 * k2 * l1;

        coeffVal += (val0 - val1 - val2 + val3) +
                    (val4 + val5 + val6 + val7) / 4.0;
    }

    return coeffVal;
}
