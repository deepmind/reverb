"""Reverb custom external dependencies."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def is_darwin(ctx):
    if ctx.os.name.lower().find("mac") != -1:
        return True
    return False

# Sanitize a dependency so that it works correctly from code that includes
# reverb as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def get_python_path(ctx):
    path = ctx.os.environ.get("PYTHON_BIN_PATH")
    if not path:
        fail(
            "Could not get environment variable PYTHON_BIN_PATH.  " +
            "Check your .bazelrc file.",
        )
    return path

def _find_tf_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_tf_lib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import tensorflow as tf; import sys; " +
            "sys.stdout.write(tf.sysconfig.get_lib())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate tensorflow installation path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_numpy_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import numpy; import sys; " +
            "sys.stdout.write(numpy.get_include())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate numpy includes path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_python_include_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "from distutils import sysconfig; import sys; " +
            "sys.stdout.write(sysconfig.get_python_inc())",
        ],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate python includes path:\n{}"
            .format(exec_result.stderr))
    return exec_result.stdout.splitlines()[-1]

def _find_python_solib_path(repo_ctx):
    exec_result = repo_ctx.execute(
        [
            get_python_path(repo_ctx),
            "-c",
            "import sys; vi = sys.version_info; " +
            "sys.stdout.write('python{}.{}'.format(vi.major, vi.minor))",
        ],
    )
    if exec_result.return_code != 0:
        fail("Could not locate python shared library path:\n{}"
            .format(exec_result.stderr))
    version = exec_result.stdout.splitlines()[-1]
    exec_result = repo_ctx.execute(
        ["{}-config".format(version), "--configdir"],
        quiet = True,
    )
    if exec_result.return_code != 0:
        fail("Could not locate python shared library path:\n{}"
            .format(exec_result.stderr))

    if is_darwin(repo_ctx):
        basename = "lib{}m.dylib".format(version)
        solib_dir = "/".join(exec_result.stdout.splitlines()[-1].split("/")[:-2])
    else:
        basename = "lib{}.so".format(version)
        solib_dir = exec_result.stdout.splitlines()[-1]

    full_path = repo_ctx.path("{}/{}".format(solib_dir, basename))
    if not full_path.exists:
        basename = basename.replace('m.dylib', '.dylib')
        full_path = repo_ctx.path("{}/{}".format(solib_dir, basename))
        if not full_path.exists:
            fail("Unable to find python shared library file:\n{}/{}"
                .format(solib_dir, basename))
    return struct(dir = solib_dir, basename = basename)

def _eigen_archive_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["tf_includes/Eigen/**/*.h",
                 "tf_includes/Eigen/**",
                 "tf_includes/unsupported/Eigen/**/*.h",
                 "tf_includes/unsupported/Eigen/**"]),
    # https://groups.google.com/forum/#!topic/bazel-discuss/HyyuuqTxKok
    includes = ["tf_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _nsync_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path + "/external", "nsync_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["nsync_includes/nsync/public/*.h"]),
    includes = ["nsync_includes"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _zlib_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/zlib",
        "zlib",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["zlib/**/*.h"]),
    includes = ["zlib"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _snappy_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(
        tf_include_path + "/external/snappy",
        "snappy",
    )
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(["snappy/*.h"]),
    includes = ["snappy"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _protobuf_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tf_includes")
    repo_ctx.symlink(Label("//third_party:protobuf.BUILD"), "BUILD")

def _tensorflow_includes_repo_impl(repo_ctx):
    tf_include_path = _find_tf_include_path(repo_ctx)
    repo_ctx.symlink(tf_include_path, "tensorflow_includes")
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "includes",
    hdrs = glob(
        [
            "tensorflow_includes/**/*.h",
            "tensorflow_includes/third_party/eigen3/**",
        ],
        exclude = ["tensorflow_includes/absl/**/*.h"],
    ),
    includes = ["tensorflow_includes"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@eigen_archive//:includes",
        "@protobuf_archive//:includes",
        "@zlib_includes//:includes",
        "@snappy_includes//:includes",
    ],
    visibility = ["//visibility:public"],
)
filegroup(
    name = "protos",
    srcs = glob(["tensorflow_includes/**/*.proto"]),
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

def _tensorflow_solib_repo_impl(repo_ctx):
    tf_lib_path = _find_tf_lib_path(repo_ctx)
    repo_ctx.symlink(tf_lib_path, "tensorflow_solib")
    if is_darwin(repo_ctx):
        suffix = "2.dylib"
    else:
        suffix = "so.2"

    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "framework_lib",
    srcs = ["tensorflow_solib/libtensorflow_framework.{suffix}"],
    deps = ["@python_includes", "@python_includes//:numpy_includes"],
    visibility = ["//visibility:public"],
)
""".format(suffix=suffix))

def _python_includes_repo_impl(repo_ctx):
    python_include_path = _find_python_include_path(repo_ctx)
    python_solib = _find_python_solib_path(repo_ctx)
    repo_ctx.symlink(python_include_path, "python_includes")
    numpy_include_path = _find_numpy_include_path(repo_ctx)
    repo_ctx.symlink(numpy_include_path, "numpy_includes")
    repo_ctx.symlink(
        "{}/{}".format(python_solib.dir, python_solib.basename),
        python_solib.basename,
    )

    if is_darwin(repo_ctx):
        # Fix Fatal Python error: PyThreadState_Get: no current thread
        python_includes_srcs = ""
    else:
        python_includes_srcs = 'srcs = ["%s"],' % python_solib.basename

    # Note, "@python_includes" is a misnomer since we include the
    # libpythonX.Y.so in the srcs, so we can get access to python's various
    # symbols at link time.
    repo_ctx.file(
        "BUILD",
        content = """
cc_library(
    name = "python_includes",
    hdrs = glob(["python_includes/**/*.h"]),
    {srcs}
    includes = ["python_includes"],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "numpy_includes",
    hdrs = glob(["numpy_includes/**/*.h"]),
    includes = ["numpy_includes"],
    visibility = ["//visibility:public"],
)
""".format(srcs=python_includes_srcs),
        executable = False,
    )

def cc_tf_configure():
    """Autoconf pre-installed tensorflow repo."""
    make_eigen_repo = repository_rule(implementation = _eigen_archive_repo_impl)
    make_eigen_repo(name = "eigen_archive")
    make_nsync_repo = repository_rule(
        implementation = _nsync_includes_repo_impl,
    )
    make_nsync_repo(name = "nsync_includes")
    make_zlib_repo = repository_rule(
        implementation = _zlib_includes_repo_impl,
    )
    make_zlib_repo(name = "zlib_includes")
    make_snappy_repo = repository_rule(
        implementation = _snappy_includes_repo_impl,
    )
    make_snappy_repo(name = "snappy_includes")
    make_protobuf_repo = repository_rule(
        implementation = _protobuf_includes_repo_impl,
    )
    make_protobuf_repo(name = "protobuf_archive")
    make_tfinc_repo = repository_rule(
        implementation = _tensorflow_includes_repo_impl,
    )
    make_tfinc_repo(name = "tensorflow_includes")
    make_tflib_repo = repository_rule(
        implementation = _tensorflow_solib_repo_impl,
    )
    make_tflib_repo(name = "tensorflow_solib")
    make_python_inc_repo = repository_rule(
        implementation = _python_includes_repo_impl,
    )
    make_python_inc_repo(name = "python_includes")

def reverb_python_deps():
    http_archive(
        name = "pybind11",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
        ],
        sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
        strip_prefix = "pybind11-2.4.3",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
    )

    http_archive(
        name = "absl_py",
        sha256 = "603febc9b95a8f2979a7bdb77d2f5e4d9b30d4e0d59579f88eba67d4e4cc5462",
        strip_prefix = "abseil-py-pypi-v0.9.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
            "https://github.com/abseil/abseil-py/archive/pypi-v0.9.0.tar.gz",
        ],
    )

def _reverb_protoc_archive(ctx):
    version = ctx.attr.version
    sha256 = ctx.attr.sha256

    override_version = ctx.os.environ.get("REVERB_PROTOC_VERSION")
    if override_version:
        sha256 = ""
        version = override_version

    if is_darwin(ctx):
        platform = "osx"
        sha256 = ""
    else:
        platform = "linux"

    urls = [
        "https://github.com/protocolbuffers/protobuf/releases/download/v%s/protoc-%s-%s-x86_64.zip" % (version, version, platform),
    ]
    ctx.download_and_extract(
        url = urls,
        sha256 = sha256,
    )

    ctx.file(
        "BUILD",
        content = """
filegroup(
    name = "protoc_bin",
    srcs = ["bin/protoc"],
    visibility = ["//visibility:public"],
)
""",
        executable = False,
    )

reverb_protoc_archive = repository_rule(
    implementation = _reverb_protoc_archive,
    attrs = {
        "version": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
    },
)

def reverb_protoc_deps(version, sha256):
    reverb_protoc_archive(name = "protobuf_protoc", version = version, sha256 = sha256)
