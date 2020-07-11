
/* util.hpp */

#ifndef UTIL_HPP
#define UTIL_HPP

#include <memory>
#include <string>

#include <opencv2/core.hpp>

/* Get mp4 video file name from Win32 file open dialog */
bool GetVideoFileName(std::string& fileNameStr);

#endif /* UTIL_HPP */
