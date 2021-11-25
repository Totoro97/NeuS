#include <cnpy.h>
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>


void Calibrate(std::string work_dir, int n_images, double raw_scale) {
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<int> board_ids;

  int n_rows = 20;
  int n_cols = 20;

  double scale = (n_cols * 8 - 1) / double(std::sqrt(2) * raw_scale);

  // Define the coordinates of markers in the board.
  for (int x = 0; x < n_rows; x++) {
    for (int y = 0; y < n_cols; y++) {
      int a = (x - n_rows / 2) * 8;
      int b = (y - n_cols / 2) * 8;

      obj_points.emplace_back();
      board_ids.emplace_back(x * n_cols + y);
      auto& current_vec = obj_points.back();
      current_vec.emplace_back(a / scale, b / scale, 0);
      current_vec.emplace_back((a + 7) / scale, b / scale, 0);
      current_vec.emplace_back((a + 7) / scale, (b + 7) / scale, 0);
      current_vec.emplace_back(a / scale, (b + 7) / scale, 0);
    }
  }

  // Create ArUco board
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_1000);
  cv::Ptr<cv::aruco::Board> board = cv::aruco::Board::create(obj_points, dictionary, board_ids);
  std::vector<std::vector<cv::Point2f>> all_corners_cated;
  std::vector<int> all_ids_cated;
  std::vector<int> marker_count_per_frame;

  cv::Size img_size;
  for (int i = 0; i < n_images; i++) {
    std::string idx = std::to_string(i);
    while (idx.length() < 3) {
        idx = "0" + idx;
    }
    cv::Mat img = cv::imread(work_dir + "/images/" + idx + ".png");
    int height = img.rows;
    int width = img.cols;
    uint8_t* data = img.data;
    for (int k = 0; k < height * width * 3; k += 3) {
        uint8_t color = 255 - data[k + 1];
        data[k] = data[k + 1] = data[k + 2] = color;
    }

    cv::imwrite(work_dir + "/tmp/image_for_aruco/" + idx + ".png", img);
    img_size = img.size();
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners, rejected_candidates;
    cv::aruco::detectMarkers(img, dictionary, marker_corners, marker_ids, parameters, rejected_candidates);
    std::cout << marker_corners.size() << " " << marker_ids.size() << std::endl;
    for (auto& corner : marker_corners) {
      all_corners_cated.emplace_back(corner);
    }
    for (auto& id : marker_ids) {
      all_ids_cated.emplace_back(id);
    }
    marker_count_per_frame.emplace_back(marker_ids.size());
  }

  cv::Mat cameraMatrix, distCoeffs;
  std::vector<cv::Mat> rvecs, tvecs;
  double repError = cv::aruco::calibrateCameraAruco(all_corners_cated, all_ids_cated, marker_count_per_frame, board, img_size, cameraMatrix, distCoeffs, rvecs, tvecs);

  std::vector<double> poses(n_images * 6);
  std::vector<double> intrinsic(9);
  for (int i = 0; i < n_images; i++) {
    for (int j = 0; j < 3; j++) {
      poses[i * 6 + j] = rvecs[i].at<double>(j, 0);
      poses[i * 6 + j + 3] = tvecs[i].at<double>(j, 0);
    }
  }
  for (int i = 0; i < 9; i++) {
    intrinsic[i] = cameraMatrix.at<double>(i / 3, i % 3);
  }


  cnpy::npy_save(work_dir + "/tmp/poses.npy", poses.data(), { (unsigned int) n_images, 2, 3 }, "w");
  cnpy::npy_save(work_dir + "/tmp/intrinsic.npy", intrinsic.data(), { 3, 3 }, "w");
  std::cout << "r_vecs: " << rvecs.size() << std::endl;
  std::cout << cameraMatrix << std::endl;
  std::cout << distCoeffs << std::endl;

  for (int i = 0; i < n_images; i++) {
    std::string idx = std::to_string(i);
    while (idx.length() < 3) {
        idx = "0" + idx;
    }
    cv::Mat img = cv::imread(work_dir + "/images/" + idx + ".png");
    cv::Mat new_img;
    cv::Mat new_mat;
    cv::undistort(img, new_img, cameraMatrix, distCoeffs, new_mat);
    cv::imwrite(work_dir + "/tmp/image_undistort/" + idx + ".png", new_img);
  }
}


int main(int argc, char* argv[]) {
  Calibrate(std::string(argv[1]), std::stoi(argv[2]), std::stod(argv[3]));
}

