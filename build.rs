use cmake;
use cxx_build;

fn main() {
    if cfg!(feature = "voro-static") {
        let dst = cmake::Config::new("extern/voro")
            .define("VORO_BUILD_EXAMPLES", "OFF")
            .define("VORO_BUILD_CMD_LINE", "OFF")
            .define("VORO_ENABLE_DOXYGEN", "OFF")
            .build();

        println!("cargo:rustc-link-search=native={}", dst.display());

        cxx_build::bridge("src/nlist/voro.rs")
            .file("src/nlist/voro.cc")
            .include(dst.join("include"))
            .flag_if_supported("-std=c++17")
            .compile("schmeud");

        println!("cargo:rustc-link-lib=static=voro++");
    } else if cfg!(feature = "voro-system") {
        cxx_build::bridge("src/nlist/voro.rs")
            .file("src/nlist/voro.cc")
            .include("-lvoro++")
            .flag_if_supported("-std=c++17")
            .compile("schmeud");

        println!("cargo:rustc-link-lib=voro++");
    } else {
        panic!("Either voro-static or voro-system must be enabled");
    }
}
