#include "gtest_mpi.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gtest_mpi {

namespace {

class MPIListener : public testing::EmptyTestEventListener {
public:
  using UnitTest = testing::UnitTest;
  using TestCase = testing::TestCase;
  using TestInfo = testing::TestInfo;
  using TestPartResult = testing::TestPartResult;
  using TestSuite = testing::TestSuite;

  MPIListener(testing::TestEventListener *listener)
      : listener_(listener), comm_(MPI_COMM_WORLD), gather_called_(false) {
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
    int rank;
    MPI_Comm_rank(comm_, &rank);
    if (rank != 0) listener_.reset();
  }

  void OnTestProgramStart(const UnitTest &u) override {
    if (listener_) listener_->OnTestProgramStart(u);
  }

  void OnTestProgramEnd(const UnitTest &u) override {
    if (listener_) listener_->OnTestProgramEnd(u);
  }

  void OnTestStart(const TestInfo &test_info) override {
    gather_called_ = false;
    if (listener_) listener_->OnTestStart(test_info);
  }

  void OnTestPartResult(const TestPartResult &test_part_result) override {
    if (listener_) {
      listener_->OnTestPartResult(test_part_result);
    } else if (test_part_result.type() == TestPartResult::Type::kFatalFailure ||
               test_part_result.type() == TestPartResult::Type::kNonFatalFailure) {
      std::size_t file_index = strings_.size();
      strings_ += test_part_result.file_name();
      strings_ += '\0';

      std::size_t message_index = strings_.size();
      strings_ += test_part_result.message();
      strings_ += '\0';

      infos_.emplace_back(ResultInfo{test_part_result.type(), file_index,
                                     test_part_result.line_number(), message_index});
    }
  }

  void OnTestEnd(const TestInfo &test_info) override {
    if (!gather_called_) {
      std::cerr << "Missing GTEST_MPI_GUARD in test case!" << std::endl;
      throw std::runtime_error("Missing GTEST_MPI_GUARD in test case!");
    }

    if (listener_) listener_->OnTestEnd(test_info);
  }

  void OnTestIterationStart(const UnitTest &u, int it) override {
    if (listener_) listener_->OnTestIterationStart(u, it);
  }

  void OnEnvironmentsSetUpStart(const UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsSetUpStart(u);
  }

  void OnEnvironmentsSetUpEnd(const UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsSetUpEnd(u);
  }

  void OnTestSuiteStart(const TestSuite &t) override {
    if (listener_) listener_->OnTestSuiteStart(t);
  }

  void OnTestDisabled(const TestInfo &t) override {
    if (listener_) listener_->OnTestDisabled(t);
  }
  void OnTestSuiteEnd(const TestSuite &t) override {
    if (listener_) listener_->OnTestSuiteEnd(t);
  }

  void OnEnvironmentsTearDownStart(const UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsTearDownStart(u);
  }

  void OnEnvironmentsTearDownEnd(const UnitTest &u) override {
    if (listener_) listener_->OnEnvironmentsTearDownEnd(u);
  }

  void OnTestIterationEnd(const UnitTest &u, int it) override {
    if (listener_) listener_->OnTestIterationEnd(u, it);
  }

  void GatherPartResults() {
    gather_called_ = true;
    int rank, n_proc;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &n_proc);

    if (rank == 0) {
      decltype(infos_) remote_infos;
      decltype(strings_) remote_strings;
      for (int r = 1; r < n_proc; ++r) {
        MPI_Status status;
        int count;

        // Result infos
        MPI_Probe(r, 0, comm_, &status);
        MPI_Get_count(&status, MPI_CHAR, &count);
        auto num_results =
            static_cast<std::size_t>(count) / sizeof(decltype(remote_infos)::value_type);
        remote_infos.resize(num_results);
        MPI_Recv(remote_infos.data(), count, MPI_BYTE, r, 0, comm_, MPI_STATUS_IGNORE);

        // Only continue if any results
        if (num_results) {
          // Get strings
          MPI_Probe(r, 0, comm_, &status);
          MPI_Get_count(&status, MPI_CHAR, &count);
          auto string_size =
              static_cast<std::size_t>(count) / sizeof(decltype(remote_strings)::value_type);
          remote_strings.resize(string_size);
          MPI_Recv(&remote_strings[0], count, MPI_BYTE, r, 0, comm_, MPI_STATUS_IGNORE);

          // Create error for every remote fail
          for (const auto &info : remote_infos) {
            if (info.type == TestPartResult::Type::kFatalFailure ||
                info.type == TestPartResult::Type::kNonFatalFailure) {
              ADD_FAILURE_AT(&remote_strings[info.file_index], info.line_number)
                  << "Rank " << r << ": " << &remote_strings[info.message_index];
            }
          }
        }
      }
    } else {
      MPI_Send(infos_.data(), infos_.size() * sizeof(decltype(infos_)::value_type), MPI_BYTE, 0, 0,
               comm_);

      // Only send string if results exist
      if (infos_.size()) {
        MPI_Send(strings_.data(), strings_.size() * sizeof(decltype(strings_)::value_type),
                 MPI_BYTE, 0, 0, comm_);
      }
    }

    infos_.clear();
    strings_.clear();
  }

private:
  struct ResultInfo {
    TestPartResult::Type type;
    std::size_t file_index;
    int line_number;
    std::size_t message_index;
  };

  std::unique_ptr<testing::TestEventListener> listener_;
  MPI_Comm comm_;
  bool gather_called_;

  std::vector<ResultInfo> infos_;
  std::string strings_;
};

MPIListener *globalMPIListener = nullptr;

}  // namespace

void InitGoogleTestMPI(int *argc, char **argv) {
  ::testing::InitGoogleTest(argc, argv);

  auto &test_listeners = ::testing::UnitTest::GetInstance()->listeners();

  globalMPIListener =
      new MPIListener(test_listeners.Release(test_listeners.default_result_printer()));

  test_listeners.Append(globalMPIListener);
}

TestGuard CreateTestGuard() {
  return TestGuard{[]() { globalMPIListener->GatherPartResults(); }};
}

}  // namespace gtest_mpi
