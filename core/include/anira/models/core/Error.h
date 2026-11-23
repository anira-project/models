// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>

namespace anira::models {

// Thrown at configuration and construction time only (a missing model file, a model that does
// not match the spec a library is written for, a refused anira call). Never on a real-time entry.
class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace anira::models
