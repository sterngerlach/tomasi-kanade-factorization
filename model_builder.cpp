
/* model_builder.cpp */

#include "model_builder.hpp"

#include <algorithm>
#include <cassert>
#include <iterator>

#include <iostream>

/* Build 3D model from point correspondences */
void ModelBuilder::Build(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& points2D,
    const std::vector<cv::Point3f>& points3D)
{
    assert(points2D.size() == points3D.size());

    /* Set an image used for texture */
    this->mImage = image;

    /* Set 2D and 3D point correspondences */
    this->mPoints2D = points2D;
    this->mPoints3D = points3D;

    /* Perform Delaunay triangulation from 2D points */
    this->mSubdiv2D.initDelaunay(cv::Rect(0, 0, image.cols, image.rows));
    this->mSubdiv2D.insert(points2D);
    this->mSubdiv2D.getTriangleList(this->mTriangles2D);

    /* Create 3D triangle meshes */
    this->mTriangles3D.reserve(this->mTriangles2D.size());

    this->mNumOfValidTriangles = 0;

    for (std::size_t i = 0; i < this->mTriangles2D.size(); ++i) {
        cv::Vec6f& tri2D = this->mTriangles2D[i];

        Triangle3D tri3D;

        /* Check if the 2D points in the 2D triangle are inside the image */
        if (tri2D[0] < 0 || tri2D[0] >= this->mImage.cols ||
            tri2D[1] < 0 || tri2D[1] >= this->mImage.rows ||
            tri2D[2] < 0 || tri2D[2] >= this->mImage.cols ||
            tri2D[3] < 0 || tri2D[3] >= this->mImage.rows ||
            tri2D[4] < 0 || tri2D[4] >= this->mImage.cols ||
            tri2D[5] < 0 || tri2D[5] >= this->mImage.rows) {
            tri3D.mValid = false;
            this->mTriangles3D.push_back(std::move(tri3D));
            continue;
        }

        /* Update the number of valid triangles */
        ++this->mNumOfValidTriangles;

        /* Get the corresponding 3D points */
        const auto pointIt0 = std::find_if(
            std::begin(points2D), std::end(points2D),
            [tri2D](const cv::Point2f& pt2D) {
                return tri2D[0] == pt2D.x && tri2D[1] == pt2D.y; });
        const auto pointIt1 = std::find_if(
            std::begin(points2D), std::end(points2D),
            [tri2D](const cv::Point2f& pt2D) {
                return tri2D[2] == pt2D.x && tri2D[3] == pt2D.y; });
        const auto pointIt2 = std::find_if(
            std::begin(points2D), std::end(points2D),
            [tri2D](const cv::Point2f& pt2D) {
                return tri2D[4] == pt2D.x && tri2D[5] == pt2D.y; });

        const std::size_t pointIdx0 =
            std::distance(std::begin(points2D), pointIt0);
        const std::size_t pointIdx1 =
            std::distance(std::begin(points2D), pointIt1);
        const std::size_t pointIdx2 =
            std::distance(std::begin(points2D), pointIt2);

        tri3D.mPoints[0] = points3D[pointIdx0];
        tri3D.mPoints[1] = points3D[pointIdx1];
        tri3D.mPoints[2] = points3D[pointIdx2];

        tri3D.mPointIndices[0] = pointIdx0;
        tri3D.mPointIndices[1] = pointIdx1;
        tri3D.mPointIndices[2] = pointIdx2;

        /* Order the 3D points counterclockwise */
        const cv::Vec3f vec01 {
            tri2D[2] - tri2D[0], tri2D[3] - tri2D[1], 0.0f };
        const cv::Vec3f vec02 {
            tri2D[4] - tri2D[0], tri2D[5] - tri2D[1], 0.0f };
        const cv::Vec3f vecNormal2D = vec01.cross(vec02);

        /* Swap the 3D points if the mesh is not facing towards the viewer */
        if (vecNormal2D[2] > 0.0f) {
            std::swap(tri2D[2], tri2D[4]);
            std::swap(tri2D[3], tri2D[5]);
            std::swap(tri3D.mPoints[1], tri3D.mPoints[2]);
            std::swap(tri3D.mPointIndices[1], tri3D.mPointIndices[2]);
        }

        /* Compute normal vectors using 3D points */
        const cv::Vec3f vec3D01 {
            tri3D.mPoints[1].x - tri3D.mPoints[0].x,
            tri3D.mPoints[1].y - tri3D.mPoints[0].y,
            tri3D.mPoints[1].z - tri3D.mPoints[0].z };
        const cv::Vec3f vec3D02 {
            tri3D.mPoints[2].x - tri3D.mPoints[0].x,
            tri3D.mPoints[2].y - tri3D.mPoints[0].y,
            tri3D.mPoints[2].z - tri3D.mPoints[0].z };
        cv::Vec3f vecNormal3D = vec3D01.cross(vec3D02);
        vecNormal3D /= cv::norm(vecNormal3D);

        tri3D.mVecNormal = vecNormal3D;
        tri3D.mValid = true;

        /* Append the 3D triangle mesh */
        this->mTriangles3D.push_back(std::move(tri3D));
    }
}

/* Convert the 3D model to Viz3D data format */
void ModelBuilder::ToViz3DMesh(cv::viz::Mesh& modelMesh)
{
    const int numOfTriangles = static_cast<int>(this->mNumOfValidTriangles);

    /* Setup mesh data */
    cv::Mat pointCloudMat { cv::Size(3 * numOfTriangles, 1), CV_32FC3 };
    cv::Mat normalsMat { cv::Size(3 * numOfTriangles, 1), CV_32FC3 };
    cv::Mat polygonsMat { cv::Size(4 * numOfTriangles, 1), CV_32SC1 };
    cv::Mat texCoordsMat { cv::Size(3 * numOfTriangles, 1), CV_32FC2 };

    for (int i = 0, j = 0; i < this->mTriangles3D.size(); ++i) {
        const Triangle3D& tri3D = this->mTriangles3D[i];

        if (!tri3D.mValid)
            continue;

        /* Set 3D point coordinates */
        pointCloudMat.at<cv::Point3f>(0, 3 * j) = tri3D.mPoints[0];
        pointCloudMat.at<cv::Point3f>(0, 3 * j + 1) = tri3D.mPoints[1];
        pointCloudMat.at<cv::Point3f>(0, 3 * j + 2) = tri3D.mPoints[2];

        /* Set normal vectors */
        normalsMat.at<cv::Vec3f>(0, 3 * j) = tri3D.mVecNormal;
        normalsMat.at<cv::Vec3f>(0, 3 * j + 1) = tri3D.mVecNormal;
        normalsMat.at<cv::Vec3f>(0, 3 * j + 2) = tri3D.mVecNormal;

        /* Set polygons (indices of the point cloud) */
        polygonsMat.at<int>(0, 4 * j) = 3;
        polygonsMat.at<int>(0, 4 * j + 1) = 3 * j;
        polygonsMat.at<int>(0, 4 * j + 2) = 3 * j + 1;
        polygonsMat.at<int>(0, 4 * j + 3) = 3 * j + 2;

        /* Set 2D texture coordinates */
        const float imageCols = static_cast<float>(this->mImage.cols);
        const float imageRows = static_cast<float>(this->mImage.rows);

        texCoordsMat.at<cv::Vec2f>(0, 3 * j)[0] =
            this->mPoints2D[tri3D.mPointIndices[0]].x / imageCols;
        texCoordsMat.at<cv::Vec2f>(0, 3 * j)[1] =
            this->mPoints2D[tri3D.mPointIndices[0]].y / imageRows;
        texCoordsMat.at<cv::Vec2f>(0, 3 * j + 1)[0] =
            this->mPoints2D[tri3D.mPointIndices[1]].x / imageCols;
        texCoordsMat.at<cv::Vec2f>(0, 3 * j + 1)[1] =
            this->mPoints2D[tri3D.mPointIndices[1]].y / imageRows;
        texCoordsMat.at<cv::Vec2f>(0, 3 * j + 2)[0] =
            this->mPoints2D[tri3D.mPointIndices[2]].x / imageCols;
        texCoordsMat.at<cv::Vec2f>(0, 3 * j + 2)[1] =
            this->mPoints2D[tri3D.mPointIndices[2]].y / imageRows;

        ++j;
    }

    /* Set mesh data */
    modelMesh.cloud = pointCloudMat;
    modelMesh.normals = normalsMat;
    modelMesh.polygons = polygonsMat;
    modelMesh.tcoords = texCoordsMat;
    modelMesh.texture = this->mImage;

    return;
}
