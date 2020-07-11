
/* model_builder.hpp */

#ifndef MODEL_BUILDER_HPP
#define MODEL_BUILDER_HPP

#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

/*
 * ModelBuilder class is for 3D reconstruction
 */
class ModelBuilder
{
public:
    /* Triangle3D struct represents a 3D triangle mesh */
    struct Triangle3D
    {
        bool        mValid;
        cv::Point3f mPoints[3];
        std::size_t mPointIndices[3];
        cv::Vec3f   mVecNormal;

        Triangle3D() : mValid(false) { }
    };

public:
    ModelBuilder() = default;
    ~ModelBuilder() = default;

    /* Build 3D model from point correspondences */
    void Build(const cv::Mat& image,
               const std::vector<cv::Point2f>& points2D,
               const std::vector<cv::Point3f>& points3D);

    /* Convert the 3D model to Viz3D data format */
    void ToViz3DMesh(cv::viz::Mesh& modelMesh);

    /* Get the 2D triangles */
    inline const std::vector<cv::Vec6f>& Triangles2D() const
    { return this->mTriangles2D; }
    /* Get the 3D triangles */
    inline const std::vector<Triangle3D>& Triangles3D() const
    { return this->mTriangles3D; }

private:
    cv::Subdiv2D             mSubdiv2D;
    cv::Mat                  mImage;
    std::vector<cv::Point2f> mPoints2D;
    std::vector<cv::Point3f> mPoints3D;
    std::vector<cv::Vec6f>   mTriangles2D;
    std::vector<Triangle3D>  mTriangles3D;
    std::size_t              mNumOfValidTriangles;
};

#endif /* MODEL_BUILDER_HPP */
