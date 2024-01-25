def make_exe():

    dist = default_python_distribution()

    policy = dist.make_python_packaging_policy()

    policy.resources_location_fallback = "filesystem-relative:prefix"

    policy.extension_module_filter = "all"

    python_config = dist.make_python_interpreter_config()

    python_config.allocator_backend = "default"

    python_config.filesystem_importer = True

    python_config.run_filename = "galaktatts.py"

    exe = dist.to_python_executable(
        name="galaktatts",

        packaging_policy=policy,

        config=python_config,
    )

    exe.windows_subsystem = "console"

    exe.add_python_resources(exe.pip_download(["pyttsx3==2.90", "SpeechRecognition==3.8.1", "pyflakes==2.2.0", "pocketsphinx==5"]))


    return exe

def make_embedded_resources(exe):
    return exe.to_embedded_resources()

def make_install(exe):

    files = FileManifest()

    files.add_python_resource(".", exe)

    return files

def make_msi(exe):

    return exe.to_wix_msi_builder(

        "galakta",

        "galakta tts",

        "1.1",

        "simplyYan"
    )

def register_code_signers():

    if not VARS.get("ENABLE_CODE_SIGNING"):
        return



register_code_signers()

register_target("exe", make_exe)
register_target("resources", make_embedded_resources, depends=["exe"], default_build_script=True)
register_target("install", make_install, depends=["exe"], default=True)
register_target("msi_installer", make_msi, depends=["exe"])

resolve_targets()