use cmake;
use cxx_build;

fn main() {
    // std::env::set_var("CXX", "clang++-14");
    if cfg!(feature = "voro-static") {
        let dst = cmake::Config::new("extern/voro")
            .define("VORO_BUILD_EXAMPLES", "OFF")
            .define("VORO_BUILD_CMD_LINE", "OFF")
            .define("VORO_ENABLE_DOXYGEN", "OFF")
            .define("CMAKE_INSTALL_PREFIX", std::env::var("OUT_DIR").unwrap())
            .define("CMAKE_INSTALL_LIBDIR", "lib")
            .build();

        println!("cargo:rustc-link-search={}", dst.join("lib").display());

        cxx_build::bridge("src/nlist/voro.rs")
            .file("src/nlist/voro.cc")
            .include(dst.join("include"))
            .flag("-lvoro++")
            .flag_if_supported("-Wno-unused-parameter")
            .flag_if_supported("-Wno-unused-variable")
            .flag_if_supported("-Wno-delete-non-abstract-non-virtual-dtor")
            .flag_if_supported("-std=c++17")
            .compile("schmeud");


        println!("cargo:rustc-link-lib=static=voro++");
    } else if cfg!(feature = "voro-system") {
        cxx_build::bridge("src/nlist/voro.rs")
            .file("src/nlist/voro.cc")
            .flag("-lvoro++")
            .flag_if_supported("-std=c++17")
            .compile("schmeud");

        println!("cargo:rustc-link-lib=voro++");
    } else {
        panic!("Either voro-static or voro-system must be enabled");
    }
}
