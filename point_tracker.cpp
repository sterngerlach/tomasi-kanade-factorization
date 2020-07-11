
/* point_tracker.cpp */

#include "point_tracker.hpp"

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video/tracking.hpp>

/* Track points from video file */
bool PointTracker::TrackPoints(const std::string& videoFileName)
{
    /* Create VideoCapture object */
    cv::VideoCapture videoCapture;
    videoCapture.open(videoFileName);

    if (!videoCapture.isOpened()) {
        std::cerr << "Failed to open video file: "
                  << videoFileName << std::endl;
        return false;
    }

    const int frameWidth = static_cast<int>(
        videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int frameHeight = static_cast<int>(
        videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

    const int scaledWidth = frameWidth / 2;
    const int scaledHeight = frameHeight / 2;

    bool isFirstFrame = true;
    std::size_t numOfFrames = 0;
    std::size_t numOfPoints = 0;

    std::vector<PointTrackNodePtr> pointTrackNodes;
    std::vector<cv::Point2f> currentFeatures;
    std::vector<cv::Point2f> prevFeatures;
    cv::Mat currentFrame;
    cv::Mat prevFrame;
    cv::Mat firstFrame;

    while (true) {
        cv::Mat frame;
        videoCapture.read(frame);

        if (frame.empty())
            break;

        /* Scale the frame */
        cv::Mat scaledFrame;
        cv::resize(frame, scaledFrame, cv::Size(scaledWidth, scaledHeight));

        /* Convert to grayscale image */
        cv::cvtColor(scaledFrame, currentFrame, cv::COLOR_BGR2GRAY);

        std::cerr << "Processing frame " << numOfFrames << '\n';

        if (isFirstFrame) {
            /* Detect good feature points to track */
            cv::goodFeaturesToTrack(
                currentFrame, currentFeatures, 500, 0.01, 5);
            prevFeatures = currentFeatures;

            /* Save the first frame for later 3D reconstruction */
            scaledFrame.copyTo(firstFrame);

            isFirstFrame = false;

            /* Update point tracks */
            for (std::size_t i = 0; i < currentFeatures.size(); ++i) {
                auto newNode = std::make_shared<PointTrackNode>(
                    nullptr, currentFeatures[i]);
                pointTrackNodes.push_back(newNode);
            }
        } else {
            /* Track feature points using Lucas-Kanade method */
            std::vector<unsigned char> statuses;
            std::vector<float> errors;
            cv::calcOpticalFlowPyrLK(
                prevFrame, currentFrame, prevFeatures, currentFeatures,
                statuses, errors);
            numOfPoints = 0;

            /* Remove points that could not be tracked */
            for (std::size_t i = 0; i < statuses.size(); ++i) {
                if (!statuses[i])
                    continue;

                prevFeatures[numOfPoints] = prevFeatures[i];
                currentFeatures[numOfPoints] = currentFeatures[i];

                auto currentNode = pointTrackNodes[i];
                auto newNode = std::make_shared<PointTrackNode>(
                    currentNode, currentFeatures[i]);
                pointTrackNodes[numOfPoints] = std::move(newNode);

                ++numOfPoints;
            }

            /* Update point tracks and feature points */
            prevFeatures.resize(numOfPoints);
            currentFeatures.resize(numOfPoints);
            pointTrackNodes.resize(numOfPoints);
        }

        ++numOfFrames;

        cv::swap(currentFrame, prevFrame);
        std::swap(currentFeatures, prevFeatures);
    }

    /* Release VideoCapture object */
    videoCapture.release();

    /* Set the first frame for later reconstruction */
    this->mFirstFrame = firstFrame;
    /* Set the size of each frame */
    this->mFrameWidth = scaledWidth;
    this->mFrameHeight = scaledHeight;
    /* Set the total number of tracked points */
    this->mNumOfPoints = pointTrackNodes.size();
    /* Set the total number of frames processed */
    this->mNumOfFrames = numOfFrames;
    
    /* Build point track information for later use */
    this->mTracks.clear();
    this->mTracks.reserve(numOfFrames);

    for (std::size_t i = 0; i < this->mNumOfPoints; ++i) {
        PointTrack pointTrack;

        for (auto trackNode = pointTrackNodes[i];
             trackNode != nullptr;
             trackNode = trackNode->mParent)
            pointTrack.push_front(trackNode->mPoint);

        assert(pointTrack.size() == this->mNumOfFrames);
        this->mTracks.push_back(std::move(pointTrack));
    }

    assert(this->mTracks.size() == this->mNumOfPoints);

    return true;
}
