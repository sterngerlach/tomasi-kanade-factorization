
/* util.cpp */

#include "util.hpp"

#include <Windows.h>

/* Get mp4 video file name from Win32 file open dialog */
bool GetVideoFileName(std::string& fileNameStr)
{
    char fileName[MAX_PATH];
    ::ZeroMemory(fileName, sizeof(fileName));

    OPENFILENAMEA openDialogParams;
    ::ZeroMemory(&openDialogParams, sizeof(openDialogParams));

    openDialogParams.lStructSize = sizeof(openDialogParams);
    openDialogParams.hwndOwner = nullptr;
    openDialogParams.lpstrFilter = "MP4 files {*.mp4}\0*.mp4\0"
                                   "All files {*.*}\0*.*\0\0";
    openDialogParams.lpstrCustomFilter = nullptr;
    openDialogParams.nMaxCustFilter = 0;
    openDialogParams.nFilterIndex = 0;
    openDialogParams.lpstrFile = fileName;
    openDialogParams.nMaxFile = MAX_PATH;
    openDialogParams.Flags = OFN_FILEMUSTEXIST;

    if (!::GetOpenFileNameA(&openDialogParams))
        return false;

    fileNameStr = fileName;
    return true;
}
