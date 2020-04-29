#pragma once

#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <windows.h>

namespace {

    wchar_t *convertCharArrayToLPCWSTR(const char* charArray)
    {
        wchar_t* wString = new wchar_t[4096];
        MultiByteToWideChar(CP_ACP, 0, charArray, -1, wString, 4096);
        return wString;
    }

    void output(const char* szFormat, ...)
    {
        char szBuff[4096];
        va_list arg;
        va_start(arg, szFormat);
        _vsnprintf_s(szBuff, sizeof(szBuff), szFormat, arg);
        va_end(arg);

        LPCWSTR string = convertCharArrayToLPCWSTR(szBuff);

        OutputDebugString(string);

        delete[] string;
    }

    void loadBinary(const wchar_t* filename, uint8_t*& data)
    {
        std::ifstream stream;
        stream.open(filename, std::ifstream::in | std::ifstream::binary);
        if (stream.good()) {
            stream.seekg(0, std::ios::end);
            size_t size = size_t(stream.tellg());
            data = new uint8_t[size];
            stream.seekg(0, std::ios::beg);
            stream.read((char*)data, size);
            stream.close();
        }
    }

    void loadData(const wchar_t* xFilename, std::vector<double*>& xs, const wchar_t* yFilename, std::vector<double*>& ys, uint32_t& elementSize)
    {
        uint8_t* imageData;
        loadBinary(xFilename, imageData);
        uint32_t imageCount = (imageData[4] << 24) | (imageData[5] << 16) | (imageData[6] << 8) | (imageData[7]);
        uint32_t imageWidth = (imageData[8] << 24) | (imageData[9] << 16) | (imageData[10] << 8) | (imageData[11]);
        uint32_t imageHeight = (imageData[12] << 24) | (imageData[13] << 16) | (imageData[14] << 8) | (imageData[15]);
        uint32_t pixelCount = imageWidth * imageHeight;
        imageData += 16;

        uint8_t* labelsData;
        loadBinary(yFilename, labelsData);
        labelsData += 8;

        xs.resize(imageCount);
        ys.resize(imageCount);

        for (uint32_t i = 0; i < imageCount; i++)
        {
            double* x = new double[pixelCount] { 0 };
            for (uint32_t j = 0; j < pixelCount; j++)
            {
                x[j] = (double)(imageData[pixelCount * i + j]) / 255.0;
            }
            xs[i] = x;

            uint8_t label = labelsData[i];
            double* y = new double[10]{ 0 };
            y[label] = 1.0;
            ys[i] = y;
        }

        elementSize = pixelCount;
    }
}