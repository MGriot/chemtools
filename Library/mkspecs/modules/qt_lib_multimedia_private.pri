QT.multimedia_private.VERSION = 5.15.6
QT.multimedia_private.name = QtMultimedia
QT.multimedia_private.module =
QT.multimedia_private.libs = $$QT_MODULE_LIB_BASE
QT.multimedia_private.includes = $$QT_MODULE_INCLUDE_BASE/QtMultimedia/5.15.6 $$QT_MODULE_INCLUDE_BASE/QtMultimedia/5.15.6/QtMultimedia
QT.multimedia_private.frameworks =
QT.multimedia_private.depends = core_private gui_private multimedia
QT.multimedia_private.uses =
QT.multimedia_private.module_config = v2 internal_module
QT.multimedia_private.enabled_features = directshow evr wasapi wmf wmsdk wshellitem
QT.multimedia_private.disabled_features = alsa gpu_vivante gstreamer_1_0 gstreamer_0_10 gstreamer gstreamer_app gstreamer_encodingprofiles gstreamer_gl gstreamer_photography linux_v4l openal pulseaudio resourcepolicy
QMAKE_LIBS_DIRECTSHOW = -lstrmiids -ldmoguids -luuid -lmsdmo -lole32 -loleaut32
QMAKE_LIBS_WMF = -lstrmiids -ldmoguids -luuid -lmsdmo -lole32 -loleaut32 -lMf -lMfuuid -lMfplat -lPropsys
