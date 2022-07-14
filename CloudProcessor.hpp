//
// Created by Aleksej on 29.04.2022.
//

#ifndef TOP_CLOUDPROCESSOR_HPP
#define TOP_CLOUDPROCESSOR_HPP

#include <pcl/common/io.h>
#include <pcl/common/transforms.h>

#include <pcl/point_cloud.h>

#include <pcl/console/time.h>

#include <pcl/io/ply_io.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/search/impl/kdtree.hpp>

#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/opencv.hpp>

#include <memory>
#include <utility>

class CloudProcessor
{
private:
    template<class T>
    static T clamp(T value, T min, T max)
    {
        return std::max(std::min(value, max), min);
    }

    static auto x_comp(const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.x < p2.x; };

    static auto y_comp(const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.y < p2.y; };

    static auto z_comp(const pcl::PointXYZRGB& p1, const pcl::PointXYZRGB& p2) { return p1.z < p2.z; };

    static bool loadCloud(const std::string& filename, pcl::PCLPointCloud2& cloud);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointCloud;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr putInBound();

    static cv::Mat removeHoles(const cv::Mat& image, int radius);

    static cv::Mat RemoveHolesWithMeans(const cv::Mat& img, int delta);

    static cv::Mat RemoveHolesWithReplaceExpanding(const cv::Mat& img);

    static pcl::PointCloud<pcl::PointXYZRGB>::Ptr ProjectToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                                                 const Eigen::Vector3f& origin,
                                                                 const Eigen::Vector3f& axis_x,
                                                                 const Eigen::Vector3f& axis_y);

    static std::list<cv::Vec3b> getPixelsInRadius(const cv::Mat& img, const cv::Point2i& point, int radius,
                                                  const std::function<bool(const cv::Vec3b&)>& pred);

public:
    explicit CloudProcessor(const std::string& ply_file);

    void visualize();

    void passThrough(float min, float max, const std::string& field);

    void radiusOutlierRemoval(float radius, int minNeighbors);

    void voxelGrid(float lx, float ly, float lz);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr extractMaxCluster();

    void reset(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc);

    void moveToBase();

    void exportRGBImage(const std::string& path, const cv::Size& imgSize, int radius);

    void exportRGBExperimental(const std::string& path);
};

CloudProcessor::CloudProcessor(const std::string& ply_file)
{
    pcl::PCLPointCloud2 loadedCow;
    if (!loadCloud(ply_file, loadedCow))
    {
        throw std::exception("Can't load cloud");
    }

    pointCloud = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

    pcl::fromPCLPointCloud2(loadedCow, *pointCloud);
}

bool CloudProcessor::loadCloud(const std::string& filename, pcl::PCLPointCloud2& cloud)
{
    using namespace pcl;
    using namespace pcl::console;
    using namespace pcl::io;

    TicToc tt;
    print_highlight("Loading ");
    print_value("%CloudProcessor ", filename.c_str());

    tt.tic();
    if (loadPLYFile(filename, cloud) < 0)
        return false;
    print_info("[done, ");
    print_value("%g", tt.toc());
    print_info(" ms : ");
    print_value("%d", cloud.width * cloud.height);
    print_info(" points]\n");
    print_info("Available dimensions: ");
    print_value("%CloudProcessor\n", pcl::getFieldsList(cloud).c_str());

    return true;
}

void CloudProcessor::visualize()
{
    pcl::visualization::PCLVisualizer viewer;

    viewer.addPointCloud(pointCloud, "cloud");
    viewer.addCoordinateSystem();
    viewer.setBackgroundColor(0, 0, 0);
    viewer.initCameraParameters();

    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }
}

void CloudProcessor::passThrough(float min, float max, const std::string& field)
{
    pcl::PassThrough<pcl::PointXYZRGB> filter;
    filter.setFilterFieldName(field);
    filter.setFilterLimits(min, max);
    filter.setInputCloud(pointCloud);
    filter.filter(*pointCloud);
}

void CloudProcessor::radiusOutlierRemoval(float radius, int minNeighbors)
{
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> filter;
    filter.setInputCloud(pointCloud);
    filter.setRadiusSearch(radius);
    filter.setMinNeighborsInRadius(minNeighbors);
    filter.filter(*pointCloud);
}

void CloudProcessor::voxelGrid(float lx, float ly, float lz)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> filter;
    filter.setInputCloud(pointCloud);
    filter.setLeafSize(lx, ly, lz);
    filter.filter(*pointCloud);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudProcessor::extractMaxCluster()
{
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    tree->setInputCloud(pointCloud);

    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
    ec.setClusterTolerance(0.02);
    ec.setSearchMethod(tree);
    ec.setInputCloud(pointCloud);
    ec.extract(clusterIndices);

    auto maxCluster = std::max_element(clusterIndices.begin(), clusterIndices.end(),
                                       [](const pcl::PointIndices& p1, const pcl::PointIndices& p2)
                                       {
                                           return p1.indices.size() < p2.indices.size();
                                       });

    if (maxCluster == clusterIndices.end())
        return {};

    return std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(*pointCloud, maxCluster->indices);
}

void CloudProcessor::reset(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc)
{
    pointCloud = std::move(pc);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudProcessor::putInBound()
{
    pcl::PointXYZRGB coloredPoint;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr bound(new pcl::PointCloud<pcl::PointXYZRGB>());

    float xMin = std::min_element(pointCloud->begin(), pointCloud->end(), x_comp)->x;
    float xMax = std::max_element(pointCloud->begin(), pointCloud->end(), x_comp)->x;
    float yMin = std::min_element(pointCloud->begin(), pointCloud->end(), y_comp)->y;
    float yMax = std::max_element(pointCloud->begin(), pointCloud->end(), y_comp)->y;
    float zMin = std::min_element(pointCloud->begin(), pointCloud->end(), z_comp)->z;
    float zMax = std::max_element(pointCloud->begin(), pointCloud->end(), z_comp)->z;

    // create parallelepiped around cloud
    for (float x = xMin; x < xMax; x += 0.01)
    {
        coloredPoint.x = x;
        coloredPoint.y = yMin;
        coloredPoint.z = zMin;
        bound->push_back(coloredPoint);
        coloredPoint.y = yMax;
        bound->push_back(coloredPoint);
        coloredPoint.z = zMax;
        bound->push_back(coloredPoint);
        coloredPoint.y = yMin;
        bound->push_back(coloredPoint);
    }
    for (float y = yMin; y < yMax; y += 0.01)
    {
        coloredPoint.x = xMin;
        coloredPoint.y = y;
        coloredPoint.z = zMin;
        bound->push_back(coloredPoint);
        coloredPoint.x = xMax;
        bound->push_back(coloredPoint);
        coloredPoint.z = zMax;
        bound->push_back(coloredPoint);
        coloredPoint.x = xMin;
        bound->push_back(coloredPoint);
    }
    for (float z = zMin; z < zMax; z += 0.01)
    {
        coloredPoint.x = xMin;
        coloredPoint.y = yMin;
        coloredPoint.z = z;
        bound->push_back(coloredPoint);
        coloredPoint.x = xMax;
        bound->push_back(coloredPoint);
        coloredPoint.y = yMax;
        bound->push_back(coloredPoint);
        coloredPoint.x = xMin;
        bound->push_back(coloredPoint);
    }

    return bound;
}

void CloudProcessor::moveToBase()
{
    auto bound = putInBound();

    float xMin = bound->begin()->x;
    float yMin = bound->begin()->y;
    float zMin = bound->begin()->z;

    for (auto& i: *bound)
    {
        xMin = std::min(xMin, i.x);
        yMin = std::min(yMin, i.y);
        zMin = std::min(zMin, i.z);
    }

    Eigen::Matrix4f first_translation(4, 4);
    first_translation << 1, 0, 0, -xMin,
            0, 1, 0, -yMin,
            0, 0, 1, -zMin,
            0, 0, 0, 1;
    pcl::transformPointCloud(*pointCloud, *pointCloud, first_translation);
}

void CloudProcessor::exportRGBImage(const std::string& path, const cv::Size& imgSize, int radius)
{
    cv::Mat image = cv::Mat::zeros(imgSize, CV_8UC3);

    float zMax = std::max_element(pointCloud->begin(), pointCloud->end(), z_comp)->z;
    float xMax = std::max_element(pointCloud->begin(), pointCloud->end(), x_comp)->x;
    float yMax = std::max_element(pointCloud->begin(), pointCloud->end(), y_comp)->y;
    std::cout << xMax << ' ' << yMax << '\n';

    double xCoefficient = (imgSize.width - 1) / 1.82;
    double yCoefficient = (imgSize.height - 1) / 0.62;
    double zCoefficient = UCHAR_MAX / zMax;

    for (auto& point: *pointCloud)
    {
        int yInd = (int) std::round(point.y * yCoefficient);
        int xInd = (int) std::round(point.x * xCoefficient);
        int zInd = (int) std::round(point.z * zCoefficient);

        auto value = clamp<int>(UCHAR_MAX - zInd, 0, UCHAR_MAX);

        if (image.at<uchar>(yInd, xInd) < value)
        {
            image.at<cv::Vec3b>(yInd, xInd) = {point.b, point.g, point.r};
        }
    }

    cv::flip(image, image, 0);

//    for (int i = 0; i < radius; ++i)
//        image = removeHoles(image, 3);
//    for (int i = 0; i < radius; ++i)
//        image = RemoveHolesWithReplaceExpanding(image);

    image = RemoveHolesWithMeans(image, radius);

    cv::imwrite(path, image);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudProcessor::ProjectToPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                                                      const Eigen::Vector3f& origin,
                                                                      const Eigen::Vector3f& axis_x,
                                                                      const Eigen::Vector3f& axis_y)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, *aux_cloud);

    auto normal = axis_x.cross(axis_y);
    Eigen::Hyperplane<float, 3> plane(normal, origin);

    for (auto& itPoint: *aux_cloud)
    {
        // project point to plane
        auto proj = plane.projection(itPoint.getVector3fMap());
        itPoint.getVector3fMap() = proj;
    }
    return aux_cloud;
}

cv::Mat CloudProcessor::removeHoles(const cv::Mat& image, int radius)
{
    static const cv::Vec3b blackPoint = {0, 0, 0};
    cv::Mat newImage;
    image.copyTo(newImage);

    for (int y = 0 + radius; y < image.rows - radius; ++y)
    {
        for (int x = 0 + radius; x < image.cols - radius; ++x)
        {
            if (image.at<cv::Vec3b>(y, x) != blackPoint)
                continue;

            auto pixels = getPixelsInRadius(image, {x, y}, radius,
                                            [](const cv::Vec3b& p)
                                            {
                                                return p != cv::Vec3b{0, 0, 0};
                                            });

            if (pixels.empty())
                continue;

            newImage.at<cv::Vec3b>(y, x) = pixels.front();
            x += radius * 2;
        }
    }

    return std::move(newImage);
}

cv::Mat CloudProcessor::RemoveHolesWithMeans(const cv::Mat& img, int delta)
{
    cv::Mat new_img;
    img.copyTo(new_img);

    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; x += 1)
        {
            auto pixels = getPixelsInRadius(img, {x, y}, delta,
                                            [](const cv::Vec3b& p)
                                            {
                                                return p != cv::Vec3b{0, 0, 0};
                                            });
            if (pixels.size() >= 5)
            {
                cv::Vec3i colors = {0, 0, 0};
                for (auto& pix: pixels)
                    colors += pix;
                for (int i = 0; i < 3; ++i)
                    colors[i] /= (int) pixels.size();
                new_img.at<cv::Vec3b>(y, x) = colors;
            }
        }
    }
    //new_img.at<cv::Vec3b>(0, 0) = {0, 0, 255};

    return new_img;
}

cv::Mat CloudProcessor::RemoveHolesWithReplaceExpanding(const cv::Mat& img)
{
    const int radius = 7;
    const cv::Vec3b blackPoint = {0, 0, 0};
    cv::Mat new_img;
    img.copyTo(new_img);

    for (int y = 0 + radius; y < img.rows - radius; ++y)
    {
        for (int x = 0 + radius; x < img.cols - radius; ++x)
        {
            if (img.at<cv::Vec3b>(y, x) != blackPoint)
                continue;
            std::list<cv::Vec3b> pixels;

            for (auto inc_radius = 3; pixels.empty() && inc_radius <= 7; inc_radius += 2)
            {
                pixels = getPixelsInRadius(img, {x, y}, inc_radius,
                                           [](const cv::Vec3b& p)
                                           {
                                               return p != cv::Vec3b{0, 0, 0};
                                           });
            }

            if (pixels.empty())
                continue;

            new_img.at<cv::Vec3b>(y, x) = pixels.front();
            x += radius * 2;
        }
    }

    return std::move(new_img);
}

std::list<cv::Vec3b> CloudProcessor::getPixelsInRadius(const cv::Mat& img, const cv::Point2i& point, int radius,
                                                       const std::function<bool(const cv::Vec3b&)>& pred)
{
    const cv::Size img_size = {img.cols, img.rows};
    std::list<cv::Vec3b> result;

    for (int iteration = 1; iteration <= radius; ++iteration)
    {
        cv::Vec2i upperLeftPoint = {point.x - iteration, point.y - iteration};
        cv::Vec2i bottomRightPoint = {point.x + iteration, point.y + iteration};

        for (int p_x = upperLeftPoint[0]; p_x < bottomRightPoint[0]; ++p_x)
        {
            if (p_x >= 0 && p_x < img_size.width)
            {

                if (upperLeftPoint[1] >= 0 && upperLeftPoint[1] < img_size.height)
                {
                    auto& p = img.at<cv::Vec3b>(upperLeftPoint[1], p_x);
                    if (pred(p))
                        result.push_back(p);
                }
                if (bottomRightPoint[1] >= 0 && bottomRightPoint[1] < img_size.height)
                {
                    auto& p = img.at<cv::Vec3b>(bottomRightPoint[1], p_x);
                    if (pred(p))
                        result.push_back(p);
                }
            }
        }

        for (int p_y = upperLeftPoint[1]; p_y < bottomRightPoint[1]; ++p_y)
        {
            if (p_y >= 0 && p_y < img_size.height)
            {
                if (upperLeftPoint[0] >= 0 && upperLeftPoint[0] < img_size.width)
                {
                    auto& p = img.at<cv::Vec3b>(p_y, upperLeftPoint[0]);
                    if (pred(p))
                        result.push_back(p);
                }
                if (bottomRightPoint[0] >= 0 && bottomRightPoint[0] < img_size.width)
                {
                    auto& p = img.at<cv::Vec3b>(p_y, bottomRightPoint[0]);
                    if (pred(p))
                        result.push_back(p);
                }
            }
        }
    }

    return std::move(result);
}

void CloudProcessor::exportRGBExperimental(const std::string& path)
{
    auto projection = ProjectToPlane(pointCloud, {0, 0, 0}, {1, 0, 0}, {0, 1, 0});

    pcl::visualization::PCLVisualizer viewer;

    viewer.addPointCloud(projection);
    viewer.setShowFPS(false);
    viewer.setBackgroundColor(0, 0, 0);
    viewer.setCameraPosition(1, 0, -2.2, 0.75, 0.5, 1.5
                             , 0, 0, 0);
    viewer.saveScreenshot(path);

    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
    }
}


#endif //TOP_CLOUDPROCESSOR_HPP
