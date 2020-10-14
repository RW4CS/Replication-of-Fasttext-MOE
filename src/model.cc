/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "model.h"
#include "loss.h"
#include "utils.h"
#include "densematrix.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace fasttext {

Model::State::State(int32_t hiddenSize, int32_t outputSize, int32_t seed)
    : lossValue_(0.0),
      nexamples_(0),
      hidden(hiddenSize),
      output(outputSize),
      grad(hiddenSize),
      rng(seed) {}

real Model::State::getLoss() const {
  return lossValue_ / nexamples_;
}

void Model::State::incrementNExamples(real loss) {
  lossValue_ += loss;
  nexamples_++;
}

Model::Model(
    std::shared_ptr<Matrix> wi,
    std::shared_ptr<Matrix> wo,
    std::shared_ptr<Loss> loss,
    bool normalizeGradient)
    : wi_(wi), wo_(wo), loss_(loss), normalizeGradient_(normalizeGradient) {}

void Model::computeHidden(const std::vector<int32_t>& input, State& state)
    const {
  Vector& hidden = state.hidden;
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    hidden.addRow(*wi_, *it);
  }
  hidden.mul(1.0 / input.size());
}

void Model::computeHidden2(const std::vector<int32_t>& input, State& state, real ratio) const {
    Vector& hidden = state.hidden;
    hidden.zero();
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        //printf("Reach here 2 \n");
        //std::cerr << *it << std::endl;
        hidden.addRow2(*wi_, *it, ratio);
    }
    //printf("Reach here 2 \n");
    hidden.mul(1.0 / input.size());
}


void Model::predict(
    const std::vector<int32_t>& input,
    int32_t k,
    real threshold,
    Predictions& heap,
    State& state) const {
  if (k == Model::kUnlimitedPredictions) {
    k = wo_->size(0); // output size
  } else if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  heap.reserve(k + 1);
  computeHidden(input, state);

  loss_->predict(k, threshold, heap, state);
}


void Model::update(
        const std::vector<int32_t>& input,
        const std::vector<int32_t>& targets,
        int32_t targetIndex,
        real lr,
        State& state) {
    if (input.size() == 0) {
        return;
    }
    computeHidden(input, state);

    Vector& grad = state.grad;
    grad.zero();
    real lossValue = loss_->forward(targets, targetIndex, state, lr, true, 1);
    state.incrementNExamples(lossValue);

    if (normalizeGradient_) {
        grad.mul(1.0 / input.size());
    }
    for (auto it = input.cbegin(); it != input.cend(); ++it) {
        wi_->addVectorToRow(grad, *it, 1.0);
    }
}


void Model::update_ori(
    const std::vector<int32_t>& input,
    const std::vector<int32_t>& targets,
    int32_t targetIndex,
    real lr,
    State& state,
    real alpha,
    real ratio,
    std::shared_ptr<Dictionary> dict) {
  if (input.size() == 0) {
    return;
  }
  computeHidden2(input, state, 1);
  //for (int32_t i = 0; i < state.hidden.size(); i++) {
  //    if (std::isnan(state.hidden[i])) {
  //        std::cerr << d << std::endl;
  //        throw std::runtime_error("Encountered NaN.");
  //    }
  //}
  Vector& grad = state.grad;
  grad.zero();
  real lossValue = loss_->forward(targets, targetIndex, state, lr, true, alpha);
  state.incrementNExamples(lossValue);
  //std::cerr << "raw loss is : " << lossValue << std::endl;

  if (normalizeGradient_) {
    grad.mul(1.0 / input.size());
  }
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    wi_->addVectorToRow(grad, *it, alpha);
  }
}


void Model::update_moe(
        const std::vector<int32_t>& ngrams,
        const std::vector<int32_t>& miswords,
        int32_t me,
        int32_t index,
        real lr,
        State& state,
        real alpha,
        real ratio,
        std::shared_ptr<Dictionary> dict) {
    if (ngrams.size() == 0) {
        return;
    }

    computeHidden2(ngrams, state, 1);
    Vector& grad = state.grad;
    grad.zero();
    real lossValue = loss_->forward_moe(miswords, index, me, state, lr, false, alpha * ratio, dict);
    state.incrementNExamples(lossValue);
    //std::cerr << "MOE loss is : " << lossValue << std::endl;
    if (normalizeGradient_) {
        grad.mul(1.0 / ngrams.size());
    }
    for (auto it = ngrams.cbegin(); it != ngrams.cend(); ++it) {
        wi_->addVectorToRow(grad, *it, 1.0);
    }
}


real Model::std_log(real x) const {
  return std::log(x + 1e-5);
}

} // namespace fasttext
