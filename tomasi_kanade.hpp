
/* tomasi_kanade.hpp */

#ifndef TOMASI_KANADE_HPP
#define TOMASI_KANADE_HPP

#include <functional>
#include <string>

#include <Eigen/Core>

#include "point_tracker.hpp"

/*
 * TomasiKanade class is for 3D reconstruction based on factorization method
 */
class TomasiKanade
{
public:
    TomasiKanade() = default;
    ~TomasiKanade() = default;

    /* Perform Tomasi-Kanade method */
    bool Run(const PointTracker& pointTracker);
    /* Perform Euclidean upgrading (orthographic) */
    bool CorrectOrthographic();
    /* Perform Euclidean upgrading (weak perspective) */
    bool CorrectWeakPerspective();
    /* Estimate camera poses (translational vectors and rotation matrices) */
    bool EstimateCameraPoseOrthographic();
    /* Estimate camera poses (translational vectors and rotation matrices) */
    bool EstimateCameraPoseWeakPerspective();

    /* Get the centroid points */
    inline const std::vector<Eigen::Vector2d>& CentroidPoints() const
    { return this->mCentroidPoints; }
    /* Get the observation matrix */
    inline const Eigen::MatrixXd& ObservationMatrix() const
    { return this->mObservationMat; }
    /* Get the motion matrix */
    inline const Eigen::MatrixXd& MotionMatrix() const
    { return this->mMotionMat; }
    /* Get the shape matrix */
    inline const Eigen::MatrixXd& ShapeMatrix() const
    { return this->mShapeMat; }

    /* Get the translation vectors */
    inline const std::vector<Eigen::Vector3d>& TranslationVectors() const
    { return this->mTranslationVectors; }
    /* Get the rotation matrices */
    inline const std::vector<Eigen::Matrix3d>& RotationMatrices() const
    { return this->mRotationMatrices; }

private:
    /* Compute symmetric coefficient matrix */
    Eigen::MatrixXd ComputeCoefficientMatrix(
        std::function<double(int, int, int, int)> coeff);
    /* Compute coefficient for Euclidean upgrading (orthographic) */
    double ComputeCoefficientOrthographic(int i, int j, int k, int l);
    /* Compute coefficient for Euclidean upgrading (weak perspective) */
    double ComputeCoefficientWeakPerspective(int i, int j, int k, int l);

private:
    PointTracker                 mPointTracker;
    std::vector<Eigen::Vector2d> mCentroidPoints;
    Eigen::MatrixXd              mObservationMat;
    Eigen::MatrixXd              mMotionMat;
    Eigen::MatrixXd              mShapeMat;

    Eigen::Matrix3d              mMetricMat;

    std::vector<Eigen::Vector3d> mTranslationVectors;
    std::vector<Eigen::Matrix3d> mRotationMatrices;

    Eigen::MatrixXd              mUMat;
    Eigen::MatrixXd              mVMat;
    Eigen::Vector3d              mSingularValuesVec;
};

#endif /* TOMASI_KANADE_HPP */
