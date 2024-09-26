
# torch_tpu 构建

```bash
cd tpu-train
source scripts/envsetup.sh sg2260 local

export SOC_CROSS_MODE=ON
export SOC_CROSS_PLATFORM=RISCV64
export CROSS_TOOLCHAINS=/work/bm_prebuilt_toolchains
export SOC_CROSS_COMPILE=1


# Debug TPU1686, optional
export EXTRA_CONFIG='-DDEBUG=ON -DUSING_FW_PRINT=ON -DUSING_FW_DEBUG=ON'

# Build TPU base kernels
rebuild_TPU1686

# Make sure we have a clean env
pip uninstall --yes torch-tpu

# Debug torch-tpu, optional
export TPUTRAIN_DEBUG=ON

# Build torch-tpu and install editable
python setup.py develop

# 构建 whl 包
python setup.py bdist_wheel
```





-DCMAKE_TOOLCHAIN_FILE=${SG1684X_TOP}/toolchain-device.cmake

-DCMAKE_TOOLCHAIN_FILE=/work/TPU1686/toolchain-device.cmake

```
      set(CMAKE_SYSTEM_NAME Linux)
      set(CMAKE_SYSTEM_PROCESSOR riscv64)
      set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
      set(CMAKE_SYSROOT ${prebuilt_path}/riscv64-linux-x86_64/sysroot)
      set(CMAKE_C_COMPILER ${prebuilt_path}/riscv64-linux-x86_64/bin/riscv64-unknown-linux-gnu-gcc)
      set(CMAKE_CXX_COMPILER ${prebuilt_path}/riscv64-linux-x86_64/bin/riscv64-unknown-linux-gnu-g++)
      set(SAFETY_FLAGS "-Wall -Wno-error=deprecated-declarations -ffunction-sections -fdata-sections -fPIC -Wno-unused-function -funwind-tables -fno-short-enums")
      if ($ENV{CHIP_ARCH} MATCHES "bm1686")
        set(arch -mcpu=c906fdv)
      else ()
        set (arch -mcpu=c920)
        add_link_options(-Wl,--allow-shlib-undefined -Wl,-dynamic-linker,/lib/ld-linux-riscv64-lp64d.so.1)
      endif()
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SAFETY_FLAGS} ${arch}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lpthread -D_GLIBCXX_USE_CXX11_ABI=1 -fno-strict-aliasing ${SAFETY_FLAGS} ${arch}")
```

## 重新构建 bmlib

增加宏定义

#define MAX_NODECHIP_NUM 1

## 构建 riscv 版本第三方库

ResourceTools.cmake 增加
```
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
        set(bfdarch "riscv:rv64")
        set(bfdname "elf64-littleriscv")
```

## TPU1686

### random_gen.cpp

/work/TPU1686/sg2260/cmodel/src/random_gen.cpp

增加

#include "stdlib.h"
#include "stdio.h"

# 交叉编译 torch


export CMAKE_BUILD_TYPE=Release
export BUILD_SHARED_LIBS=ON
export CMAKE_TOOLCHAIN_FILE=/work/torch/pytorch/riscv_linux_setup.cmake
export USE_MKLDNN=OFF
export USE_QNNPACK=OFF
export USE_PYTORCH_QNNPACK=OFF
export BUILD_TEST=OFF
export USE_NNPACK=OFF
export CAFFE2_CUSTOM_PROTOC_EXECUTABLE=/work/torch/pytorch/build_host_protoc/bin/protoc
export PYTHON_EXECUTABLE=which python3

cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_TOOLCHAIN_FILE=/work/torch/pytorch/riscv_linux_setup.cmake -DUSE_MKLDNN=OFF -DUSE_QNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DBUILD_TEST=OFF -DUSE_NNPACK=OFF -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=/work/torch/pytorch/build_host_protoc/bin/protoc -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install-riscv ../pytorch



cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_TOOLCHAIN_FILE=/work/torch/pytorch-v2.1.0/riscv_linux_setup.cmake -DUSE_MKLDNN=OFF -DUSE_QNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DBUILD_TEST=OFF -DUSE_NNPACK=OFF -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE=/work/torch/pytorch-v2.1.0/build_host_protoc/bin/protoc -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DPYTHON_INCLUDE_DIR=/work/bm_prebuilt_toolchains/build-python/include/python3.10 -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install-riscv ../pytorch-v2.1.0

增加环境变量 有些 CMAKE 参数无法通过环境变量设置需要根据报错信息手动执行
 - 头文件 ： 直接复制

？ setup.py文件增加 cmake 参数

export USE_CUDA=0 # RISC-V架构服务器无法使用CUDA
export USE_MKLDNN=0 # 并非英特尔处理器，故不支持MKL
export MAX_JOBS=32 # 编译进程数，根据自己实际需求进行更改

// export USE_DISTRIBUTED=0 # 不支持分布

删除 sleef ———— 最新版本可能已经支持

# 安装

修改 _C.cpython-310-riscv64-linux-gnu.so

## 设置 lib 路径

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/torch/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/torch_tpu/lib

or

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/torch/lib:/path/torch_tpu/lib

## ld-linux-riscv64v0p7_xthead-lp64d.so

缺少两个 so 文件，将交叉工具链中的两个文件cp到 /lib



# O

tpuDNN libtpudnn.so libsccl.so

oneDNN libdnnl.so


    spec 包含一系列头文件，定义了和架构相关的宏及inline函数，如LOCAL_MEM_START_ADDR, 寄存器定义等
    tv_gen tv_gen程序代码
    cmodel cmodel的具体实现代码
    firmware_base 包含以下内容，大部分是设备寄存器配置相关 src/atomic atomic_xxx_gen_cmd定义, 对应于指令集文档的原子指令配置 src/firmware firmware_main入口,涉及到使能TPU设置，需要操作寄存器 src/fullnet multi_fullnet实现（静态网络运行代码，里面包含了寄存器的直接配置） src/kernel tpu kernel的底层封装实现代码，不同架构不一样
    test cmodel和gen_cmd测试代码
    config_common.cmake 设备相关配置，如指定NPU_NUM_SHIFT，LOCAL_MEM_ADDR_WIDTH等
    base.cmake 与外层CMakeLists.txt通信文件，编译相关模块，并定义了BASE_LIBS变量，供外层CMakeLists.txt引用
    CMakeLists.txt 编译tv_gen与test应用
    firmware_dyn 是firmware动态加载机制里与MCU系统交互的底层函数定义
    firmware_top 是firmware裸机程序主函数入口及基础库

 ┌─────────────────────────────────────────────┐             ┌───────────────────────────────┐
 │                                             │             │                               │
 │                                             │             │                               │
 │              ┌───────────────────────────┐  │             │  ┌─────────────────────────┐  │
 │              │       Torch-Ops           │  │             │  │       Torch-TPU         │  │
 │              └────┬─────────────────┬────┘  │             │  └─────────┬───────────────┘  │
 │                   │                 │       │             │            │                  │
 │           ┌───────┼─────────────────┼───────┘             │            │                  │
 │           │       │                 │                     │            │                  │
 │           │       v                 v                     │            v                  │
 │ Torch-TPU │  ┌────────────┐  ┌───────────┐                │  ┌─────────────────────────┐  │
 │           │  │   BMLib    │  │  TPUV7Rt  │                │  │       tpuDNN            │  │
 │           │  └────┬───────┘  └──────┬────┘     ────────>  │  └───┬────────────────┬────┘  │
 │           │       │                 │                     │      │                │       │
 │           └───────┼─────────────────┼───────┐             │      │                │       │
 │                   │                 │       │             │      │                │       │
 │                   │                 │       │             │      │                │       │
 │                   v                 v       │             │      v                v       │
 │              ┌───────────────────────────┐  │             │  ┌─────────┐   ┌───────────┐  │
 │              │          firmware         │  │             │  │  BMLib  │   │  TPUV7Rt  │  │
 │              └────┬─────────────────┬────┘  │             │  └───┬─────┘   └──────┬────┘  │
 │                   │                 │       │             │      │                │       │
 └───────────────────┼─────────────────┼───────┘             │      v                v       │
                     │                 │                     │  ┌─────────────────────────┐  │
                     │                 │                     │  │       TPU1686           │  │
                     v                 v                     │  └─────────────────────────┘  │
                ┌───────────────────────────┐                │                               │
                │          TPU1686          │                │                               │
                └───────────────────────────┘                └───────────────────────────────┘
                
                
                
# Other

BMLib https://doc.sophgo.com/docs/2.7.0/docs_latest_release/bmlib/html/index.html


