#ifndef OUTPUT_KLDIVERGENCELOSS_H_
#define OUTPUT_KLDIVERGENCELOSS_H_

#include <Eigen/Core>
#include <stdexcept>
#include "../Config.h"

namespace MiniDNN
{

///
/// \ingroup Outputs
///
/// KL Divergence loss layer for soft target probabilities
///
class KLDivergenceLoss : public Output
{
private:
    Matrix m_din; // Derivative of the input of this layer
    Scalar m_loss_value; // Computed loss value

public:
    // Skip target validation
    void check_target_data(const Matrix& target) override
    {
        // Bypass validation, as KL Divergence accepts soft probabilities
    }

    void check_target_data(const IntegerVector& target) override
    {
        // This layer does not support integer targets
        throw std::invalid_argument("[class KLDivergenceLoss]: Integer target data is not supported.");
    }

    // Compute loss and gradients for soft target probabilities
    void evaluate(const Matrix& prev_layer_data, const Matrix& target) override
    {
        // Check dimensions
        const int nobs = prev_layer_data.cols();
        const int nclass = prev_layer_data.rows();

        if ((target.cols() != nobs) || (target.rows() != nclass))
        {
            throw std::invalid_argument("[class KLDivergenceLoss]: Target data have incorrect dimension.");
        }

        // Compute KL Divergence loss and its gradient
        m_din.resize(nclass, nobs);
        m_din.setZero();
        m_loss_value = Scalar(0);

        for (int i = 0; i < prev_layer_data.cols(); ++i)
        {
            for (int j = 0; j < prev_layer_data.rows(); ++j)
            {
                double prediction_val = prev_layer_data(j, i) + 1e-8; // Avoid zero
                double target_val = target(j, i) + 1e-8; // Avoid zero

                if (prediction_val <= 0 || target_val <= 0)
                {
                    throw std::invalid_argument("[class KLDivergenceLoss]: Invalid values in KL Divergence.");
                    // std::cerr << "Invalid values in KL Divergence: "
                    //           << "prev_layer_data(" << j << ", " << i << ") = " << prev_layer_data(j, i)
                    //           << ", target(" << j << ", " << i << ") = " << target(j, i) << std::endl;
                    // std::abort();
                }

                m_loss_value += target_val * std::log(target_val / prediction_val);
                m_din(j, i) = -target_val / prediction_val;
            }
        }

        m_loss_value /= nobs;
        m_din /= nobs;
    }

    // Integer targets are not supported
    void evaluate(const Matrix& prev_layer_data, const IntegerVector& target) override
    {
        throw std::invalid_argument("[class KLDivergenceLoss]: Integer target data is not supported.");
    }

    // Provide backpropagation data
    const Matrix& backprop_data() const override
    {
        return m_din;
    }

    // Return the computed loss
    Scalar loss() const override
    {
        return m_loss_value;
    }

    // Identify the layer type
    std::string output_type() const override
    {
        return "KLDivergenceLoss";
    }
};

} // namespace MiniDNN

#endif /* OUTPUT_KLDIVERGENCELOSS_H_ */

