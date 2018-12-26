/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

/*!
 *  Copyright (c) 2016 by Contributors
 * \file crop-inl.h
 * \brief the image crop operator implementation
 */

#ifndef MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_
#define MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_


#include <algorithm>
#include <vector>

#include "mxnet/base.h"
#include "dmlc/optional.h"

#include "../mxnet_op.h"
#include "../operator_common.h"
#include "image_utils.h"

namespace mxnet {
namespace op {
namespace image {

struct CenterCropParam : public dmlc::Parameter<CenterCropParam> {
  nnvm::Tuple<int> size;
  int interp;
  DMLC_DECLARE_PARAMETER(CenterCropParam) {
    DMLC_DECLARE_FIELD(size)
    .set_default(nnvm::Tuple<int>())
    .describe("Size of output image. Could be (width, height) or (size)");
    DMLC_DECLARE_FIELD(interp)
    .describe("Interpolation method for resizing. By default uses bilinear"
        "interpolation. See OpenCV's resize function for available choices.");
  }
};

bool CenterCropShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape> *in_attrs,
                             std::vector<TShape> *out_attrs) {
  // input attrs should only be (h, w, c) or (n, h, w, c)
  CHECK((in_attrs->at(0).ndim() == 3U) || (in_attrs->at(0).ndim() == 4U))
    << "Input image dimension should be 3 or 4 but got "
    << in_attrs->at(0).ndim();
  const auto& ishape = (*in_attrs)[0];
  const CenterCropParam& param = nnvm::get<CenterCropParam>(attrs.parsed);
  const auto size = GetHeightAndWidthFromSize(param);

  CHECK((size.height > 0) && (size.width > 0))
      << "Input height and width must be greater than 0";
  if (ishape.ndim() == 3) {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({size.height, size.width, ishape[C]}));
  } else {
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, TShape({ishape[N], size.height, size.width, ishape[kC]}));
  }

  return true;
}

void CenterCrop(const nnvm::NodeAttrs &attrs,
                   const OpContext &ctx,
                   const std::vector<TBlob> &inputs,
                   const std::vector<OpReqType> &req,
                   const std::vector<TBlob> &outputs) {
  CHECK_EQ(outputs.size(), 1U);
  CHECK((inputs[0].ndim() == 3) || (inputs[0].ndim() == 4))
      << "Input data must be (h, w, c) or (n, h, w, c)";
  const CenterCropParam& param = nnvm::get<CenterCropParam>(attrs.parsed);
  const auto size = GetHeightAndWidthFromSize(param);

  CenterCropImpl(inputs, outputs, size, param.interp);
  
}
}  // namespace image
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_IMAGE_CENTER_CROP_INL_H_
