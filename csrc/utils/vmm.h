// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/xpu/XPUFunctions.h>
#include <pybind11/pybind11.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace xpu_vmm {
namespace py = pybind11;
namespace vm = sycl::ext::oneapi::experimental;

// All reservations and allocations use the same SYCL context as torch kernels.
inline const sycl::context& context() { return c10::xpu::get_device_context(); }

struct Allocation {
  std::optional<vm::physical_mem> memory;
};

inline void register_bindings(py::module_& parent) {
  auto m = parent.def_submodule("vmm");
  py::class_<Allocation>(m, "Allocation");
  m.attr("dlpack_device_type") = 14;  // kDLOneAPI
  m.def("ensure_context", [](int device) {
    c10::xpu::check_device_index(device);
    if (!c10::xpu::get_raw_device(device).has(
            sycl::aspect::ext_oneapi_virtual_mem)) {
      throw std::runtime_error(
          "XPU device does not support SYCL virtual memory");
    }
  });
  m.def("unusable_runtime_reason", [](int) { return py::none(); });
  m.def("granularity", [](int device) {
    auto mode = vm::granularity_mode::minimum;
    // Both the virtual reservation and device-local mappings must be aligned.
    return std::lcm(
        vm::get_mem_granularity(context(), mode),
        vm::get_mem_granularity(
            c10::xpu::get_raw_device(device), context(), mode));
  });
  m.def("reserve", [](size_t size) {
    return vm::reserve_virtual_mem(size, context());
  });
  m.def("free_reserved", [](uintptr_t ptr, size_t size) {
    vm::free_virtual_mem(ptr, size, context());
  });
  m.def(
      "create",
      [](size_t size, int device, int export_handle_types) {
        if (export_handle_types != 0) {
          throw std::runtime_error(
              "XPU virtual memory does not export IPC handles");
        }
        return Allocation{vm::physical_mem(
            c10::xpu::get_raw_device(device), context(), size)};
      },
      py::arg("size"),
      py::arg("device_index"),
      py::arg("export_handle_types") = 0);
  m.def("map", [](uintptr_t ptr, size_t size, const Allocation& handle) {
    handle.memory.value().map(ptr, size, vm::address_access_mode::read_write);
  });
  // SYCL establishes access on each mapping, including disjoint segments.
  m.def("set_access", [](uintptr_t, size_t, int) {});
  m.def("unmap", [](uintptr_t ptr, size_t size) {
    vm::unmap(reinterpret_cast<void*>(ptr), size, context());
  });
  m.def("release", [](Allocation& handle) { handle.memory.reset(); });
}
}  // namespace xpu_vmm
