
/* main.cpp */

#include <cassert>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Windows.h>
#include <tchar.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz.hpp>

#include <Eigen/Core>
#include <Eigen/SVD>

#if defined(_DEBUG)
#pragma comment(lib, "opencv_world430d.lib")
#else
#pragma comment(lib, "opencv_world430.lib")
#endif

#include "model_builder.hpp"
#include "tomasi_kanade.hpp"
#include "util.hpp"

int main(int argc, char** argv)
{
    ::SetProcessDPIAware();
   
    /* Open video file */
    std::string videoFileName;

    if (!GetVideoFileName(videoFileName))
        return EXIT_SUCCESS;

    /* Track points from video file */
    PointTracker pointTracker;
    pointTracker.TrackPoints(videoFileName);

    const auto& pointTracks = pointTracker.Tracks();

    /* Perform Tomasi-Kanade method */
    TomasiKanade tomasiKanade;
    std::cerr << "Performing Tomasi-Kanade method...\n";
    tomasiKanade.Run(pointTracker);

    /* Correct motion matrix and shape matrix */
    std::cerr << "Performing Euclidean upgrading...\n";
    tomasiKanade.CorrectOrthographic();

    /* Estimate camera poses (translations and rotations) */
    tomasiKanade.EstimateCameraPoseOrthographic();

    /* Retrieve shape matrix and motion matrix */
    const Eigen::MatrixXd& shapeMat = tomasiKanade.ShapeMatrix();
    const Eigen::MatrixXd& motionMat = tomasiKanade.MotionMatrix();

    /* Setup 2D and 3D points */
    std::vector<cv::Point2f> points2D;
    points2D.reserve(shapeMat.cols());
    std::vector<cv::Point3f> points3D;
    points3D.reserve(shapeMat.cols());

    for (int i = 0; i < shapeMat.cols(); ++i) {
        /* Append the 3D point */
        const cv::Point3f point3D {
            static_cast<float>(shapeMat(0, i)),
            static_cast<float>(shapeMat(1, i)),
            static_cast<float>(shapeMat(2, i)) };
        points3D.push_back(std::move(point3D));

        /* Compute the corresponding pixel point */
        /* const Eigen::MatrixXd motionMat0 = motionMat.block<2, 3>(0, 0);
        const auto& centroidPoints = tomasiKanade.CentroidPoints();
        const Eigen::Vector2d featurePoint =
            motionMat0 * shapeMat.col(i) + centroidPoints[0];
        const cv::Point2f point2D {
            static_cast<float>(featurePoint.x()),
            static_cast<float>(featurePoint.y()) }; */
        /* Just set the tracked point */
        const cv::Point2f point2D {
            pointTracks[i][0].x, pointTracks[i][0].y };
        points2D.push_back(std::move(point2D));
    }

    /* Create 3D model */
    const cv::Mat& firstFrame = pointTracker.FirstFrame();
    ModelBuilder modelBuilder;
    modelBuilder.Build(firstFrame, points2D, points3D);

    /* Show 2D results (tracked points and triangle meshes) */
    cv::Mat featurePointsFrame;
    firstFrame.copyTo(featurePointsFrame);

    /* Draw triangle meshes */
    const auto& triangles2D = modelBuilder.Triangles2D();

    for (std::size_t i = 0; i < triangles2D.size(); ++i) {
        const cv::Point2f point0 { triangles2D[i](0), triangles2D[i](1) };
        const cv::Point2f point1 { triangles2D[i](2), triangles2D[i](3) };
        const cv::Point2f point2 { triangles2D[i](4), triangles2D[i](5) };
        const cv::Scalar color { 255, 0, 0 };

        cv::line(featurePointsFrame, point0, point1, color, 1, cv::LINE_AA);
        cv::line(featurePointsFrame, point1, point2, color, 1, cv::LINE_AA);
        cv::line(featurePointsFrame, point2, point0, color, 1, cv::LINE_AA);
    }

    /* Draw tracked points */
    for (std::size_t i = 0; i < points2D.size(); ++i)
        cv::circle(featurePointsFrame, points2D[i],
                    2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);

    cv::imshow("Tracked Points", featurePointsFrame);

    /* Show 3D reconstruction results */
    cv::viz::Viz3d vizResults { "3D Reconstruction Result" };

    /* Draw 3D model */
    cv::viz::Mesh modelMesh;
    modelBuilder.ToViz3DMesh(modelMesh);
    vizResults.showWidget("3D Model", cv::viz::WMesh(modelMesh));

    /* Draw 3D points */
    for (std::size_t i = 0; i < points3D.size(); ++i) {
        const cv::viz::WSphere pointSphere {
            points3D[i], 2.0, 10, cv::viz::Color::red() };
        const std::string id = "Points-" + std::to_string(i);
        vizResults.showWidget(id, pointSphere);
    }

    /* Draw camera translation vectors */
    const auto& translationVectors = tomasiKanade.TranslationVectors();

    for (std::size_t i = 0; i < translationVectors.size(); ++i) {
        const cv::Point3f point3D {
            static_cast<float>(translationVectors[i].x()),
            static_cast<float>(translationVectors[i].y()),
            static_cast<float>(translationVectors[i].z()) };
        const cv::viz::WSphere pointSphere {
            point3D, 5.0, 10, cv::viz::Color::green() };
        const std::string id = "Cameras-" + std::to_string(i);
        vizResults.showWidget(id, pointSphere);
    }

    /* Draw axes */
    vizResults.showWidget("Coordinate",
                          cv::viz::WCoordinateSystem(100.0));

    vizResults.spin();

    /* Destroy OpenCV windows */
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
