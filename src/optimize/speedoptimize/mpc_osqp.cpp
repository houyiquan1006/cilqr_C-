/*
 * Copyright (C) 2022 by SenseTime Group Limited. All rights reserved.
 *
 */

#include "mpc_osqp.hpp"

namespace senseAD {
namespace perception {
namespace speed {
MpcOsqp::MpcOsqp() {}

MpcOsqp::~MpcOsqp() {
    // Cleanup
    osqp_cleanup(osqp_workspace_);
    // clean up data and settings
    if (osqp_data_) {
        if (osqp_data_->P) csc_spfree(osqp_data_->P);
        if (osqp_data_->A) csc_spfree(osqp_data_->A);
        c_free(osqp_data_);
    }
    if (osqp_settings_) c_free(osqp_settings_);
}

void MpcOsqp::CalculateCostFunction(std::vector<c_float> *P_data,
                                    std::vector<c_int> *P_indices,
                                    std::vector<c_int> *P_indptr) {
    // 0.5*x'*P*x+q'*x
    // calcualte q
    for (size_t i = 0; i < mpc_.horizon + 1; ++i) {
        gradient_.block(i * mpc_.state_dim, 0, mpc_.state_dim, 1) =
            -1.0 * mpc_.cf_x.at(i).transpose() * mpc_.q.at(i) *
            mpc_.x_ref.at(i);
    }

    for (size_t i = 0; i < mpc_.horizon; ++i) {
        gradient_.block(
            i * mpc_.control_dim + mpc_.state_dim * (mpc_.horizon + 1), 0,
            mpc_.control_dim, 1) = -1.0 * mpc_.r.at(i) * mpc_.u_ref.at(i);
    }

    // calculate P
    int ind_p = 0;
    int p_size = osqp_state_dim_;
    P_indptr->reserve(osqp_state_dim_ + 1);
    P_data->reserve(p_size);
    P_indices->reserve(p_size);
    P_indptr->emplace_back(ind_p);
    Eigen::MatrixXd q_tmp(mpc_.state_dim, mpc_.state_dim);
    // state and terminal state
    for (size_t i = 0; i < mpc_.horizon + 1; ++i) {
        q_tmp = mpc_.cf_x.at(i).transpose() * mpc_.q.at(i) * mpc_.cf_x.at(i);
        for (size_t j = 0; j < mpc_.state_dim; ++j) {
            P_data->emplace_back(q_tmp(j, j));                // val
            P_indices->emplace_back(i * mpc_.state_dim + j);  // row
            ++ind_p;
            P_indptr->emplace_back(ind_p);
        }
    }
    // control
    const size_t state_total_dim = mpc_.state_dim * (mpc_.horizon + 1);
    for (size_t i = 0; i < mpc_.horizon; ++i) {
        for (size_t j = 0; j < mpc_.control_dim; ++j) {
            P_data->emplace_back(mpc_.r.at(i)(j, j));  // val
            P_indices->emplace_back(state_total_dim + i * mpc_.control_dim +
                                    j);  // row
            ++ind_p;
            P_indptr->emplace_back(ind_p);
        }
    }

    return;
}

void MpcOsqp::CalculateConstraintMatrix(std::vector<c_float> *A_data,
                                        std::vector<c_int> *A_indices,
                                        std::vector<c_int> *A_indptr) {
    // calculate constraint matrix A
    int ind_A = 0;
    int A_size =
        (1 + mpc_.state_dim + mpc_.state_constraint_dim) * mpc_.state_dim +
        (mpc_.discretize_order + mpc_.state_dim + mpc_.state_constraint_dim) *
            mpc_.state_dim * (mpc_.horizon - mpc_.discretize_order) +
        (mpc_.discretize_order + mpc_.state_constraint_dim) * mpc_.state_dim +
        (mpc_.state_dim + mpc_.control_constraint_dim) * mpc_.control_dim *
            mpc_.horizon;
    for (size_t i = 1; i < mpc_.discretize_order; ++i) {
        A_size +=
            (i + mpc_.state_dim + mpc_.state_constraint_dim) * mpc_.state_dim;
    }

    A_indptr->reserve(osqp_state_dim_ + 1);
    A_data->reserve(A_size);
    A_indices->reserve(A_size);
    A_indptr->emplace_back(ind_A);
    size_t constraint_row;

    for (size_t j = 0; j < mpc_.state_dim; ++j) {
        constraint_row = j;
        A_data->emplace_back(-1);                 // value
        A_indices->emplace_back(constraint_row);  // row
        ++ind_A;
        for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
            constraint_row = mpc_.state_dim + k;
            A_data->emplace_back(mpc_.ad.at(0)(k, j));  // value
            A_indices->emplace_back(constraint_row);    // row
            ++ind_A;
        }
        for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {
            constraint_row = k + mpc_.state_dim * (mpc_.horizon + 1);
            A_data->emplace_back(mpc_.c_x.at(0)(k, j));  // value
            A_indices->emplace_back(constraint_row);     // row
            ++ind_A;
        }
        A_indptr->emplace_back(ind_A);
    }

    if (1 == mpc_.discretize_order) {
        for (size_t i = 1; i < mpc_.horizon; ++i) {        // row
            for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
                constraint_row = i * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 1](1, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                    constraint_row = (i + 1) * mpc_.state_dim + k;
                    A_data->emplace_back(mpc_.ad.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);    // row
                    ++ind_A;
                }
                for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                    constraint_row = i * mpc_.state_constraint_dim + k +
                                     mpc_.state_dim * (mpc_.horizon + 1);
                    A_data->emplace_back(mpc_.c_x.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);     // row
                    ++ind_A;
                }
                A_indptr->emplace_back(ind_A);
            }
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {
            // constraint_col = mpc_.horizon * mpc_.state_dim + j;
            constraint_row = mpc_.horizon * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 1](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {
                constraint_row = mpc_.horizon * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(mpc_.horizon)(k, j));  // value
                A_indices->emplace_back(constraint_row);                // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }
    } else if (2 == mpc_.discretize_order) {
        for (size_t j = 0; j < mpc_.state_dim; ++j) {
            constraint_row = mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[0](1, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = 2 * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(1)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {
                constraint_row = 1 * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(1)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t i = 2; i < mpc_.horizon; ++i) {        // row
            for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
                constraint_row = (i - 1) * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 2](2, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                constraint_row = i * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 1](1, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                    constraint_row = (i + 1) * mpc_.state_dim + k;
                    A_data->emplace_back(mpc_.ad.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);    // row
                    ++ind_A;
                }
                for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                    constraint_row = i * mpc_.state_constraint_dim + k +
                                     mpc_.state_dim * (mpc_.horizon + 1);
                    A_data->emplace_back(mpc_.c_x.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);     // row
                    ++ind_A;
                }
                A_indptr->emplace_back(ind_A);
            }
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (mpc_.horizon - 1) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 2](2, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = mpc_.horizon * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 1](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = mpc_.horizon * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(mpc_.horizon)(k, j));  // value
                A_indices->emplace_back(constraint_row);                // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }
    } else if (3 == mpc_.discretize_order) {
        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = 1 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[0](1, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = (1 + 1) * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(1)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = 1 * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(1)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (2 - 1) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[0](2, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            constraint_row = 2 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[1](1, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = (2 + 1) * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(2)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = 2 * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(2)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t i = 3; i < mpc_.horizon - 1; ++i) {    // row
            for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
                constraint_row = (i - 2) * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 3](3, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                constraint_row = (i - 1) * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 2](2, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                constraint_row = i * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 1](1, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                    constraint_row = (i + 1) * mpc_.state_dim + k;
                    A_data->emplace_back(mpc_.ad.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);    // row
                    ++ind_A;
                }
                for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                    constraint_row = i * mpc_.state_constraint_dim + k +
                                     mpc_.state_dim * (mpc_.horizon + 1);
                    A_data->emplace_back(mpc_.c_x.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);     // row
                    ++ind_A;
                }
                A_indptr->emplace_back(ind_A);
            }
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (mpc_.horizon - 3) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 4](3, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 2) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 3](2, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 1) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 2](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = mpc_.horizon * mpc_.state_dim + k;
                A_data->emplace_back(
                    mpc_.ad.at(mpc_.horizon - 1)(k, j));  // value
                A_indices->emplace_back(constraint_row);  // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row =
                    (mpc_.horizon - 1) * mpc_.state_constraint_dim + k +
                    mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(
                    mpc_.c_x.at(mpc_.horizon - 1)(k, j));  // value
                A_indices->emplace_back(constraint_row);   // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (mpc_.horizon - 2) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 3](3, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 1) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 2](2, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = mpc_.horizon * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 1](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = mpc_.horizon * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(mpc_.horizon)(k, j));  // value
                A_indices->emplace_back(constraint_row);                // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }
    } else if (4 == mpc_.discretize_order) {
        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = 1 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[0](1, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = 2 * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(1)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = 1 * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(1)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = 1 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[0](2, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            constraint_row = 2 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[1](1, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = 3 * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(2)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = 2 * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(2)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = 1 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[0](3, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            constraint_row = 2 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[1](2, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            constraint_row = 3 * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[2](1, 0));  // value
            A_indices->emplace_back(constraint_row);       // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = 4 * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(3)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = 3 * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(3)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t i = 4; i < mpc_.horizon - 2; ++i) {    // row
            for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
                constraint_row = (i - 3) * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 4](4, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                constraint_row = (i - 2) * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 3](3, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                constraint_row = (i - 1) * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 2](2, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                constraint_row = i * mpc_.state_dim + j;
                A_data->emplace_back(-mpc_.coeff_A[i - 1](1, 0));  // value
                A_indices->emplace_back(constraint_row);           // row
                ++ind_A;
                for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                    constraint_row = (i + 1) * mpc_.state_dim + k;
                    A_data->emplace_back(mpc_.ad.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);    // row
                    ++ind_A;
                }
                for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                    constraint_row = i * mpc_.state_constraint_dim + k +
                                     mpc_.state_dim * (mpc_.horizon + 1);
                    A_data->emplace_back(mpc_.c_x.at(i)(k, j));  // value
                    A_indices->emplace_back(constraint_row);     // row
                    ++ind_A;
                }
                A_indptr->emplace_back(ind_A);
            }
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (mpc_.horizon - 5) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 6](4, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 4) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 5](3, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 3) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 4](2, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 2) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 3](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = (mpc_.horizon - 1) * mpc_.state_dim + k;
                A_data->emplace_back(
                    mpc_.ad.at(mpc_.horizon - 2)(k, j));  // value
                A_indices->emplace_back(constraint_row);  // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row =
                    (mpc_.horizon - 2) * mpc_.state_constraint_dim + k +
                    mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(
                    mpc_.c_x.at(mpc_.horizon - 2)(k, j));  // value
                A_indices->emplace_back(constraint_row);   // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (mpc_.horizon - 4) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 5](4, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 3) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 4](3, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 2) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 3](2, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 1) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 2](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = mpc_.horizon * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.ad.at(mpc_.horizon - 1)(k, j));
                A_indices->emplace_back(constraint_row);  // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row =
                    (mpc_.horizon - 1) * mpc_.state_constraint_dim + k +
                    mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(mpc_.horizon - 1)(k, j));
                A_indices->emplace_back(constraint_row);  // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }

        for (size_t j = 0; j < mpc_.state_dim; ++j) {  // col
            constraint_row = (mpc_.horizon - 3) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 4](4, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 2) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 3](3, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = (mpc_.horizon - 1) * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 2](2, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            constraint_row = mpc_.horizon * mpc_.state_dim + j;
            A_data->emplace_back(-mpc_.coeff_A[mpc_.horizon - 1](1, 0));
            A_indices->emplace_back(constraint_row);  // row
            ++ind_A;
            for (size_t k = 0; k < mpc_.state_constraint_dim; ++k) {  // row
                constraint_row = mpc_.horizon * mpc_.state_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_x.at(mpc_.horizon)(k, j));  // value
                A_indices->emplace_back(constraint_row);                // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }
    }

    for (size_t i = 0; i < mpc_.horizon; ++i) {
        for (size_t j = 0; j < mpc_.control_dim; ++j) {    // col
            for (size_t k = 0; k < mpc_.state_dim; ++k) {  // row
                constraint_row = (i + 1) * mpc_.state_dim + k;
                A_data->emplace_back(mpc_.bd.at(i)(k, j));  // value
                A_indices->emplace_back(constraint_row);    // row
                ++ind_A;
            }
            for (size_t k = 0; k < mpc_.control_constraint_dim; ++k) {  // row
                constraint_row = i * mpc_.control_constraint_dim + k +
                                 mpc_.state_dim * (mpc_.horizon + 1) +
                                 mpc_.state_constraint_dim * (mpc_.horizon + 1);
                A_data->emplace_back(mpc_.c_u.at(i)(k, j));  // value
                A_indices->emplace_back(constraint_row);     // row
                ++ind_A;
            }
            A_indptr->emplace_back(ind_A);
        }
    }

    return;
}

void MpcOsqp::CalculateConstraintBound() {
    // evaluate the lower and the upper equality vectors
    Eigen::VectorXd lower_equality =
        Eigen::MatrixXd::Zero(mpc_.state_dim * (mpc_.horizon + 1), 1);
    lower_equality.block(0, 0, mpc_.state_dim, 1) = -1 * mpc_.x_initial;
    for (size_t i = 0; i < mpc_.horizon; ++i) {
        lower_equality.block(mpc_.state_dim * (i + 1), 0, mpc_.state_dim, 1) =
            -1.0 * mpc_.cd.at(i);
    }
    Eigen::VectorXd upper_equality;
    upper_equality = lower_equality;
    lower_equality = lower_equality;

    // evaluate the lower and the upper inequality vectors
    Eigen::VectorXd lower_inequality = Eigen::MatrixXd::Zero(
        mpc_.control_constraint_dim * mpc_.horizon +
            mpc_.state_constraint_dim * (mpc_.horizon + 1),
        1);
    Eigen::VectorXd upper_inequality = Eigen::MatrixXd::Zero(
        mpc_.control_constraint_dim * mpc_.horizon +
            mpc_.state_constraint_dim * (mpc_.horizon + 1),
        1);
    // control constraints
    for (size_t i = 0; i < mpc_.horizon; ++i) {
        lower_inequality.block(
            mpc_.control_constraint_dim * i +
                mpc_.state_constraint_dim * (mpc_.horizon + 1),
            0, mpc_.control_constraint_dim, 1) = mpc_.u_lower.at(i);
        upper_inequality.block(
            mpc_.control_constraint_dim * i +
                mpc_.state_constraint_dim * (mpc_.horizon + 1),
            0, mpc_.control_constraint_dim, 1) = mpc_.u_upper.at(i);
    }
    // state constraints
    for (size_t i = 0; i < mpc_.horizon + 1; ++i) {
        lower_inequality.block(mpc_.state_constraint_dim * i, 0,
                               mpc_.state_constraint_dim, 1) =
            mpc_.x_lower.at(i);
        upper_inequality.block(mpc_.state_constraint_dim * i, 0,
                               mpc_.state_constraint_dim, 1) =
            mpc_.x_upper.at(i);
    }

    // merge inequality and equality vectors
    // cppcheck-suppress *
    lower_bound_ << lower_equality, lower_inequality;
    // cppcheck-suppress *
    upper_bound_ << upper_equality, upper_inequality;

    return;
}

bool MpcOsqp::CheckMatrixRowsCols(const std::vector<Eigen::MatrixXd> &matrix,
                                  size_t vec_size,
                                  size_t matrix_rows,
                                  size_t matrix_cols) {
    if (matrix.size() > 0 && vec_size >= matrix.size() &&
        matrix_rows == matrix.at(0).rows() &&
        matrix_cols == matrix.at(0).cols()) {
        return true;
    }
    return false;
}

bool MpcOsqp::CheckInputOK(const MpcOsqp::Input &input) {
    // check input
    if (input.state_dim <= 0 || input.control_dim <= 0 ||
        input.state_constraint_dim <= 0 || input.control_constraint_dim <= 0 ||
        input.state_cost_function_dim <= 0 || input.horizon <= 0 ||
        input.max_iter <= 0) {
        return false;
    }

    if (input.x_initial.rows() != input.state_dim ||
        input.x_initial.cols() != 1) {
        return false;
    }

    if (input.discretize_order <= 0 || input.discretize_order > 4) {
        return false;
    }

    bool rtn = true;
    rtn &= CheckMatrixRowsCols(input.ad, input.horizon, input.state_dim,
                               input.state_dim);
    rtn &= CheckMatrixRowsCols(input.bd, input.horizon, input.state_dim,
                               input.control_dim);
    rtn &= CheckMatrixRowsCols(input.cd, input.horizon, input.state_dim, 1);
    rtn &= CheckMatrixRowsCols(input.coeff_A, input.horizon + 1,
                               input.discretize_order + 1, 1);
    rtn &= CheckMatrixRowsCols(input.q, input.horizon + 1,
                               input.state_cost_function_dim,
                               input.state_cost_function_dim);
    rtn &= CheckMatrixRowsCols(input.r, input.horizon, input.control_dim,
                               input.control_dim);
    rtn &= CheckMatrixRowsCols(input.x_ref, input.horizon + 1,
                               input.state_cost_function_dim, 1);
    rtn &=
        CheckMatrixRowsCols(input.u_ref, input.horizon, input.control_dim, 1);
    rtn &= CheckMatrixRowsCols(input.c_x, input.horizon + 1,
                               input.state_constraint_dim, input.state_dim);
    rtn &= CheckMatrixRowsCols(input.c_u, input.horizon,
                               input.control_constraint_dim, input.control_dim);
    rtn &= CheckMatrixRowsCols(input.u_lower, input.horizon,
                               input.control_constraint_dim, 1);
    rtn &= CheckMatrixRowsCols(input.u_upper, input.horizon,
                               input.control_constraint_dim, 1);
    rtn &= CheckMatrixRowsCols(input.x_lower, input.horizon + 1,
                               input.state_constraint_dim, 1);
    rtn &= CheckMatrixRowsCols(input.x_upper, input.horizon + 1,
                               input.state_constraint_dim, 1);
    rtn &= CheckMatrixRowsCols(input.cf_x, input.horizon + 1,
                               input.state_cost_function_dim, input.state_dim);
    if (rtn) {
        // copy input
        // memcpy(&mpc_, &input, sizeof(MpcOsqp::Input));
        mpc_ = input;
        osqp_state_dim_ = (input.horizon + 1) * input.state_dim +
                          input.horizon * input.control_dim;
        osqp_constraint_dim_ =
            (input.horizon + 1) * input.state_dim +
            (input.horizon + 1) * input.state_constraint_dim +
            input.horizon * input.control_constraint_dim;
    } else {
        return false;
    }

    return true;
}

// osqp error and status values
// 4 OSQP_DUAL_INFEASIBLE_INACCURATE
// 3 OSQP_PRIMAL_INFEASIBLE_INACCURATE
// 2 OSQP_SOLVED_INACCURATE
// 1 OSQP_SOLVED
// -2 OSQP_MAX_ITER_REACHED
// -3 OSQP_PRIMAL_INFEASIBLE, primal infeasible
// -4 OSQP_DUAL_INFEASIBLE, dual infeasible
// -5 OSQP_SIGINT, interrupted by user
// -6 OSQP_TIME_LIMIT_REACHED
// -7 OSQP_NON_CVX, problem non convex
// -10 OSQP_UNSOLVED, Unsolved. Only setup function has been called
MpcOsqp::Status MpcOsqp::CheckSolveStatus() {
    int status = static_cast<int>(osqp_workspace_->info->status_val);
    if (status < 0 || (status > 1)) {
        is_last_solved_ = false;
        if (!OSQPHint::StateErrorTab.count(status)) {
            std::cerr << "osqp error: unknown states, check StateErrorTab!";
            return Status::ERROR_OSQP_STATUS;
        }
        std::cerr << "osqp status error: "
                  << OSQPHint::StateErrorTab.at(status);
        return Status::ERROR_OSQP_STATUS;
    } else if (osqp_workspace_->solution == nullptr) {
        is_last_solved_ = false;
        std::cerr << "osqp solution error";
        return Status::ERROR_OSQP_SOLUTION;
    }
    // check solution nan
    for (size_t i = 0; i < osqp_state_dim_; ++i) {
        if (std::isnan(osqp_workspace_->solution->x[i])) {
            is_last_solved_ = false;
            std::cerr << "osqp solution has nan";
            return Status::ERROR_OUTPUT_NAN;
        }
    }
    is_last_solved_ = true;
    return Status::SUCCESS;
}

void MpcOsqp::InitializeMatrix() {
    gradient_ = Eigen::MatrixXd(osqp_state_dim_, 1);
    lower_bound_ = Eigen::MatrixXd(osqp_constraint_dim_, 1);
    upper_bound_ = Eigen::MatrixXd(osqp_constraint_dim_, 1);

    return;
}

void MpcOsqp::Data() {
    osqp_data_->n = osqp_state_dim_;
    osqp_data_->m = osqp_constraint_dim_;

    // 0.5*x'*P*x+q'*x
    // calculate cost function matrix and gradient vector
    P_data_.clear();
    P_indices_.clear();
    P_indptr_.clear();
    CalculateCostFunction(&P_data_, &P_indices_, &P_indptr_);
    osqp_data_->P = csc_matrix(osqp_state_dim_, osqp_state_dim_, P_data_.size(),
                               CopyData(P_data_), CopyData(P_indices_),
                               CopyData(P_indptr_));
    osqp_data_->q = gradient_.data();

    // l<=A*x<=u
    // calcualte constraint
    A_data_.clear();
    A_indices_.clear();
    A_indptr_.clear();
    CalculateConstraintMatrix(&A_data_, &A_indices_, &A_indptr_);
    osqp_data_->A = csc_matrix(osqp_constraint_dim_, osqp_state_dim_,
                               A_data_.size(), CopyData(A_data_),
                               CopyData(A_indices_), CopyData(A_indptr_));

    // calculate lower_bound_ and upper_bound_
    CalculateConstraintBound();
    osqp_data_->l = lower_bound_.data();
    osqp_data_->u = upper_bound_.data();

    return;
}

void MpcOsqp::Settings() {
    // default setting
    osqp_set_default_settings(osqp_settings_);
    osqp_settings_->polish = true;
    osqp_settings_->scaled_termination = true;
    osqp_settings_->verbose = debug_print_ ? 1 : 0;
    osqp_settings_->max_iter = mpc_.max_iter;
    osqp_settings_->eps_abs = mpc_.eps_abs;
    osqp_settings_->eps_rel = mpc_.eps_rel;
    osqp_settings_->eps_prim_inf = mpc_.eps_prim_inf;

    // osqp_settings_->adaptive_rho = false;
    // osqp_settings_->adaptive_rho_interval = 1.0;
    // osqp_settings_->check_termination = 0;
    return;
}

void MpcOsqp::Update() {
    // calculate cost function matrix
    P_data_.clear();
    P_indices_.clear();
    P_indptr_.clear();
    CalculateCostFunction(&P_data_, &P_indices_, &P_indptr_);

    // calculate constraint matrix
    A_data_.clear();
    A_indices_.clear();
    A_indptr_.clear();
    CalculateConstraintMatrix(&A_data_, &A_indices_, &A_indptr_);

    // calculate lower_bound_ and upper_bound_
    CalculateConstraintBound();

    // update
    osqp_update_lin_cost(osqp_workspace_, gradient_.data());
    osqp_update_P_A(osqp_workspace_, P_data_.data(), OSQP_NULL, P_data_.size(),
                    A_data_.data(), OSQP_NULL, A_data_.size());
    osqp_update_bounds(osqp_workspace_, lower_bound_.data(),
                       upper_bound_.data());
    osqp_update_max_iter(osqp_workspace_, mpc_.max_iter);
    osqp_update_eps_abs(osqp_workspace_, mpc_.eps_abs);
    osqp_update_eps_rel(osqp_workspace_, mpc_.eps_rel);
    osqp_update_eps_prim_inf(osqp_workspace_, mpc_.eps_prim_inf);

    return;
}

MpcOsqp::Status MpcOsqp::Init() {
    // initialize matrix
    InitializeMatrix();

    // caculate matrix
    osqp_data_ = reinterpret_cast<OSQPData *>(c_malloc(sizeof(OSQPData)));
    if (osqp_data_ == nullptr) {
        return Status::ERROR_NULL_PTR;
    }
    Data();
    // setup settings
    osqp_settings_ =
        reinterpret_cast<OSQPSettings *>(c_malloc(sizeof(OSQPSettings)));
    if (osqp_settings_ == nullptr) {
        return Status::ERROR_NULL_PTR;
    }
    Settings();

    // setup osqp work space
    c_int setup_error =
        osqp_setup(&osqp_workspace_, osqp_data_, osqp_settings_);
    if (0 != setup_error) {
        // OSQP_DATA_VALIDATION_ERROR = 1,
        // OSQP_SETTINGS_VALIDATION_ERROR = 2,
        // OSQP_LINSYS_SOLVER_LOAD_ERROR = 3,
        // OSQP_LINSYS_SOLVER_INIT_ERROR = 4,
        // OSQP_NONCVX_ERROR = 5,
        // OSQP_MEM_ALLOC_ERROR = 6,
        // OSQP_WORKSPACE_NOT_INIT_ERROR = 7,
        if (!OSQPHint::SetupErrorTab.count(static_cast<int>(setup_error))) {
            std::cerr << "osqp error: check SetupErrorTab!";
            return Status::ERROR_SETUP;
        }
        std::cerr << "osqp setup error: "
                  << OSQPHint::SetupErrorTab.at(setup_error);
        return Status::ERROR_SETUP;
    }

    return Status::SUCCESS;
}

MpcOsqp::Status MpcOsqp::Solve(const MpcOsqp::Input &input,
                               const bool debug_print,
                               std::vector<double> *output) {
    debug_print_ = debug_print;
    // check input
    if (false == CheckInputOK(input)) {
        std::cerr << "osqp input error";
        return Status::ERROR_INPUT;
    }

    output->assign(osqp_state_dim_, 0.0);

    // if not setup, then init first, the solve
    if (false == is_setup_) {
        MpcOsqp::Status init_status;
        init_status = Init();
        if (MpcOsqp::Status::SUCCESS != init_status) {
            std::cerr << "osqp init error";
            return init_status;
        }
        is_setup_ = true;
        osqp_solve(osqp_workspace_);
        Status solve_status = CheckSolveStatus();
        // output
        for (size_t i = 0; i < osqp_state_dim_; ++i) {
            output->at(i) = osqp_workspace_->solution->x[i];
        }
        return solve_status;
    }

    // warm start if last solution available
    if (is_last_solved_) {
        osqp_warm_start(osqp_workspace_, osqp_workspace_->solution->x,
                        osqp_workspace_->solution->y);
    } else {
        osqp_workspace_->settings->warm_start = false;
    }

    // update
    Update();

    // solve the problem
    osqp_solve(osqp_workspace_);

    // check status
    Status solve_status = CheckSolveStatus();

    // output
    for (size_t i = 0; i < osqp_state_dim_; ++i) {
        output->at(i) = osqp_workspace_->solution->x[i];
    }

    return solve_status;
}
}  // namespace speed
}  // namespace perception
}  // namespace senseAD
