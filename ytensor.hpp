#pragma once
/***************
* @file: ytensor.hpp
* @brief: 所有include文件的总入口。
* @author: SnifferCaptain
* @date: 2026-1-10
* @version 0.3
* @email: 3586554865@qq.com
***************/

#include "include/ytensor_concepts.hpp"
#include "include/ytensor_infos.hpp"

/////////// ytensor class def ////////////
#include "include/ytensor_base.hpp"
#include "include/ytensor_core.hpp"

//////////// external /////////////
#include "include/ytensor_function.hpp"
#include "include/ytensor_io.hpp"

//////////// implementation /////////////
#include "src/ytensor_base.inl"
#include "src/ytensor_base_math.inl"

#include "src/ytensor_core.inl"
#include "src/ytensor_math.inl"

#include "src/ytensor_io.inl"
#include "src/ytensor_function.inl"