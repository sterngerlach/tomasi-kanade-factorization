
/* point_tracker.hpp */

#ifndef POINT_TRACKER_HPP
#define POINT_TRACKER_HPP

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

/*
 * PointTrackNode struct represents the 3D position of the tracked point
 */
struct PointTrackNode
{
    PointTrackNode(std::shared_ptr<PointTrackNode> parentNode,
                   const cv::Point2f& trackedPoint) :
        mParent(parentNode),
        mPoint(trackedPoint) { }
    ~PointTrackNode() = default;

    std::shared_ptr<PointTrackNode> mParent;
    cv::Point2f                     mPoint;
};

/* Type definition for convenience */
using PointTrackNodePtr = std::shared_ptr<PointTrackNode>;
using PointTrack = std::deque<cv::Point2f>;

/*
 * PointTracker class is for tracking feature points from video
 */
class PointTracker
{
public:
    PointTracker() = default;
    ~PointTracker() = default;

    /* Track points from video file */
    bool TrackPoints(const std::string& videoFileName);

    /* Get the point tracks */
    inline const std::vector<PointTrack>& Tracks() const
    { return this->mTracks; }
    /* Get the first frame */
    inline const cv::Mat& FirstFrame() const { return this->mFirstFrame; }
    /* Get the frame width */
    inline int FrameWidth() const { return this->mFrameWidth; }
    /* Get the frame height */
    inline int FrameHeight() const { return this->mFrameHeight; }
    /* Get the number of tracked points */
    inline std::size_t NumOfPoints() const { return this->mNumOfPoints; }
    /* Get the number of frames */
    inline std::size_t NumOfFrames() const { return this->mNumOfFrames; }

private:
    std::vector<PointTrack> mTracks;
    cv::Mat                 mFirstFrame;
    int                     mFrameWidth;
    int                     mFrameHeight;
    std::size_t             mNumOfPoints;
    std::size_t             mNumOfFrames;
};

#endif /* POINT_TRACKER_HPP */
