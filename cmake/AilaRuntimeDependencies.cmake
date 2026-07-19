include_guard(GLOBAL)

function(aila_configure_runtime_dependencies)
    if(NOT WIN32)
        return()
    endif()

    get_target_property(dnnl_dll_location DNNL::dnnl IMPORTED_LOCATION_RELEASE)
    if(NOT dnnl_dll_location)
        get_target_property(dnnl_dll_location DNNL::dnnl IMPORTED_LOCATION)
    endif()

    get_filename_component(compiler_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(compiler_version_root "${compiler_bin_dir}/.." ABSOLUTE)
    get_filename_component(compiler_version "${compiler_version_root}" NAME)
    get_filename_component(compiler_component_root "${compiler_version_root}/.." ABSOLUTE)
    get_filename_component(oneapi_install_root "${compiler_component_root}/.." ABSOLUTE)

    # CMake may retain the oneAPI `latest` compiler alias. Resolve the shared
    # runtime directory from the actual compiler version in that case.
    set(toolkit_version "${compiler_version}")
    if(toolkit_version STREQUAL "latest")
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" toolkit_version
            "${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    set(oneapi_shared_bin_dir "${oneapi_install_root}/${toolkit_version}/bin")
    file(GLOB oneapi_shared_hwloc_dlls LIST_DIRECTORIES FALSE
        "${oneapi_shared_bin_dir}/libhwloc-*.dll")
    if(NOT IS_DIRECTORY "${oneapi_shared_bin_dir}" OR NOT oneapi_shared_hwloc_dlls)
        message(FATAL_ERROR
            "Resolved oneAPI shared runtime directory is invalid: ${oneapi_shared_bin_dir}")
    endif()

    find_package(TBB CONFIG REQUIRED)
    get_filename_component(tbb_root "${TBB_DIR}/../../.." ABSOLUTE)
    set(tbb_bin_dir "${tbb_root}/bin")
    if(NOT EXISTS "${tbb_bin_dir}/tbb12.dll")
        message(FATAL_ERROR "Resolved TBB runtime directory is invalid: ${tbb_bin_dir}")
    endif()

    set(runtime_source_dirs
        "${oneapi_shared_bin_dir}"
        "${compiler_bin_dir}"
        "${tbb_bin_dir}"
    )
    if(dnnl_dll_location)
        get_filename_component(dnnl_dll_dir "${dnnl_dll_location}" DIRECTORY)
        list(APPEND runtime_source_dirs "${dnnl_dll_dir}")
    endif()

    get_filename_component(linker_bin_dir "${CMAKE_LINKER}" DIRECTORY)
    if(EXISTS "${linker_bin_dir}/msvcp140.dll")
        get_filename_component(msvc_dir "${linker_bin_dir}/../../../../../.." ABSOLUTE)
        get_filename_component(msvc_tools_version "${linker_bin_dir}/../../.." NAME)
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" msvc_version_prefix "${msvc_tools_version}")
        if(msvc_version_prefix)
            file(GLOB msvc_redist_dirs
                "${msvc_dir}/Redist/MSVC/${msvc_version_prefix}*/x64/Microsoft.VC*.CRT")
        endif()
        if(NOT msvc_redist_dirs)
            file(GLOB msvc_redist_dirs
                "${msvc_dir}/Redist/MSVC/*/x64/Microsoft.VC*.CRT")
        endif()
        if(msvc_redist_dirs)
            list(SORT msvc_redist_dirs)
            list(REVERSE msvc_redist_dirs)
            list(GET msvc_redist_dirs 0 msvc_runtime_dir)
        else()
            set(msvc_runtime_dir "${linker_bin_dir}")
        endif()
        list(APPEND runtime_source_dirs "${msvc_runtime_dir}")
    endif()
    list(REMOVE_DUPLICATES runtime_source_dirs)

    aila_select_runtime_dlls(
        OUTPUT_FILES selected_runtime_dlls
        DIRECTORIES ${runtime_source_dirs}
    )

    set(runtime_search_dirs)
    set(overlay_source_dlls)
    set(lookup_dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_dependency_lookup")
    foreach(runtime_source_dir IN LISTS runtime_source_dirs)
        file(GLOB runtime_source_dlls LIST_DIRECTORIES FALSE
            "${runtime_source_dir}/*.dll")
        list(SORT runtime_source_dlls CASE INSENSITIVE)
        set(selected_from_dir)
        foreach(runtime_source_dll IN LISTS runtime_source_dlls)
            list(FIND selected_runtime_dlls "${runtime_source_dll}" selected_index)
            if(NOT selected_index EQUAL -1)
                list(APPEND selected_from_dir "${runtime_source_dll}")
            endif()
        endforeach()

        list(LENGTH runtime_source_dlls source_count)
        list(LENGTH selected_from_dir selected_count)
        if(source_count GREATER 0 AND selected_count EQUAL source_count)
            list(APPEND runtime_search_dirs "${runtime_source_dir}")
        elseif(selected_count GREATER 0)
            list(APPEND overlay_source_dlls ${selected_from_dir})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES runtime_search_dirs)
    list(REMOVE_DUPLICATES overlay_source_dlls)

    if(overlay_source_dlls)
        set(lookup_stamp "${lookup_dir}/.ready")
        set(prepare_lookup_template [=[
set(source_dlls [==[@OVERLAY_SOURCE_DLLS@]==])
set(lookup_dir [==[@LOOKUP_DIR@]==])
set(stamp_path [==[@LOOKUP_STAMP@]==])
file(REMOVE_RECURSE "${lookup_dir}")
file(MAKE_DIRECTORY "${lookup_dir}")
foreach(source_dll IN LISTS source_dlls)
    if(NOT EXISTS "${source_dll}")
        message(FATAL_ERROR "Selected runtime DLL no longer exists: ${source_dll}")
    endif()
    get_filename_component(source_name "${source_dll}" NAME)
    execute_process(
        COMMAND "@CMAKE_COMMAND@" -E copy_if_different
            "${source_dll}" "${lookup_dir}/${source_name}"
        COMMAND_ERROR_IS_FATAL ANY
    )
endforeach()
file(WRITE "${stamp_path}" "ready\n")
]=])
        set(OVERLAY_SOURCE_DLLS "${overlay_source_dlls}")
        set(LOOKUP_DIR "${lookup_dir}")
        set(LOOKUP_STAMP "${lookup_stamp}")
        string(CONFIGURE "${prepare_lookup_template}" prepare_lookup_script @ONLY)
        file(GENERATE
            OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/prepare_runtime_dependency_lookup.cmake"
            CONTENT "${prepare_lookup_script}"
        )
        add_custom_command(
            OUTPUT "${lookup_stamp}"
            COMMAND "${CMAKE_COMMAND}" -P
                "${CMAKE_CURRENT_BINARY_DIR}/prepare_runtime_dependency_lookup.cmake"
            DEPENDS
                "${CMAKE_CURRENT_BINARY_DIR}/prepare_runtime_dependency_lookup.cmake"
                ${overlay_source_dlls}
            COMMENT "Preparing conflict-free runtime DLL lookup directory"
            VERBATIM
        )
        add_custom_target(AilaRuntimeDependencyLookup
            DEPENDS "${lookup_stamp}")
        list(APPEND runtime_search_dirs "${lookup_dir}")
    endif()

    set(runtime_root_dlls)
    foreach(selected_runtime_dll IN LISTS selected_runtime_dlls)
        get_filename_component(selected_runtime_name "${selected_runtime_dll}" NAME)
        string(TOLOWER "${selected_runtime_name}" selected_runtime_key)
        if(NOT selected_runtime_key MATCHES
                "^(ur_loader|ur_win_proxy_loader|ur_adapter_opencl|ur_adapter_level_zero|ur_adapter_level_zero_v2|umf|xptifw|sycl-jit|opencl|intelocl64|common_clang64|libmmd.*|libhwloc-.*|tbb12|tbbbind|tbbbind_2_0|tbbbind_2_5|tbbmalloc|tbbmalloc_proxy|tcm|concrt140|msvcp140.*|vccorlib140|vcruntime140.*)\\.dll$")
            continue()
        endif()
        get_filename_component(selected_runtime_dir "${selected_runtime_dll}" DIRECTORY)
        list(FIND runtime_search_dirs "${selected_runtime_dir}" direct_dir_index)
        if(direct_dir_index EQUAL -1)
            list(APPEND runtime_root_dlls
                "${lookup_dir}/${selected_runtime_name}")
        else()
            list(APPEND runtime_root_dlls "${selected_runtime_dll}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES runtime_root_dlls)

    set(copy_runtime_template [=[
set(target_path "${TARGET_PATH}")
set(target_dir "${TARGET_DIR}")
set(target_kind "${TARGET_KIND}")
set(search_dirs [==[@RUNTIME_SEARCH_DIRS@]==])
set(extra_root_dlls [==[@RUNTIME_ROOT_DLLS@]==])

set(search_dll_names)
foreach(search_dir IN LISTS search_dirs)
    if(NOT IS_DIRECTORY "${search_dir}")
        message(FATAL_ERROR "Runtime dependency search directory does not exist: ${search_dir}")
    endif()
    file(GLOB search_dlls LIST_DIRECTORIES FALSE "${search_dir}/*.dll")
    foreach(search_dll IN LISTS search_dlls)
        get_filename_component(search_name "${search_dll}" NAME)
        string(TOLOWER "${search_name}" search_key)
        list(FIND search_dll_names "${search_key}" duplicate_index)
        if(NOT duplicate_index EQUAL -1)
            message(FATAL_ERROR
                "Runtime dependency search directories contain duplicate basename '${search_name}'.")
        endif()
        list(APPEND search_dll_names "${search_key}")
    endforeach()
endforeach()

set(common_dependency_args
    DIRECTORIES ${search_dirs}
    PRE_EXCLUDE_REGEXES "api-ms-win-.*" "ext-ms-.*"
    POST_EXCLUDE_REGEXES ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\].*"
)
set(resolved_deps)
set(unresolved_deps)
if(target_kind STREQUAL "EXECUTABLE")
    file(GET_RUNTIME_DEPENDENCIES
        EXECUTABLES "${target_path}"
        RESOLVED_DEPENDENCIES_VAR target_resolved_deps
        UNRESOLVED_DEPENDENCIES_VAR target_unresolved_deps
        ${common_dependency_args}
    )
elseif(target_kind STREQUAL "LIBRARY")
    file(GET_RUNTIME_DEPENDENCIES
        LIBRARIES "${target_path}"
        RESOLVED_DEPENDENCIES_VAR target_resolved_deps
        UNRESOLVED_DEPENDENCIES_VAR target_unresolved_deps
        ${common_dependency_args}
    )
else()
    message(FATAL_ERROR "Unsupported target kind: ${target_kind}")
endif()
list(APPEND resolved_deps ${target_resolved_deps})
list(APPEND unresolved_deps ${target_unresolved_deps})
if(extra_root_dlls)
    file(GET_RUNTIME_DEPENDENCIES
        LIBRARIES ${extra_root_dlls}
        RESOLVED_DEPENDENCIES_VAR extra_resolved_deps
        UNRESOLVED_DEPENDENCIES_VAR extra_unresolved_deps
        ${common_dependency_args}
    )
    list(APPEND resolved_deps ${extra_root_dlls} ${extra_resolved_deps})
    list(APPEND unresolved_deps ${extra_unresolved_deps})
endif()
list(REMOVE_DUPLICATES resolved_deps)
foreach(dep IN LISTS resolved_deps)
    if(EXISTS "${dep}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${dep}" "${target_dir}"
            COMMAND_ERROR_IS_FATAL ANY
        )
    endif()
endforeach()
if(unresolved_deps)
    list(SORT unresolved_deps)
    list(REMOVE_DUPLICATES unresolved_deps)
    message(FATAL_ERROR "Unresolved runtime dependencies for ${target_path}: ${unresolved_deps}")
endif()
]=])
    set(RUNTIME_SEARCH_DIRS "${runtime_search_dirs}")
    set(RUNTIME_ROOT_DLLS "${runtime_root_dlls}")
    string(CONFIGURE "${copy_runtime_template}" copy_runtime_script @ONLY)
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        CONTENT "${copy_runtime_script}"
    )

    function(aila_copy_runtime_dependencies target target_kind)
        if(TARGET AilaRuntimeDependencyLookup)
            add_dependencies(${target} AilaRuntimeDependencyLookup)
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                "-DTARGET_PATH=$<TARGET_FILE:${target}>"
                "-DTARGET_DIR=$<TARGET_FILE_DIR:${target}>"
                "-DTARGET_KIND=${target_kind}"
                -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
            COMMENT "Copying runtime DLLs for ${target}"
            VERBATIM
        )
    endfunction()

    if(TARGET AilaAliaRealSmoke)
        aila_copy_runtime_dependencies(AilaAliaRealSmoke EXECUTABLE)
    endif()
    if(TARGET AilaCommandSubmissionBench)
        aila_copy_runtime_dependencies(AilaCommandSubmissionBench EXECUTABLE)
    endif()
    if(TARGET AilaShared)
        aila_copy_runtime_dependencies(AilaShared LIBRARY)
    endif()

    set(release_bin_dir "${CMAKE_BINARY_DIR}/Release/bin")
    set(release_targets)
    foreach(target IN ITEMS AilaShared AilaAliaRealSmoke)
        if(TARGET ${target})
            list(APPEND release_targets ${target})
        endif()
    endforeach()
    set(release_copy_commands)
    foreach(target IN LISTS release_targets)
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL "SHARED_LIBRARY")
            set(target_kind LIBRARY)
        else()
            set(target_kind EXECUTABLE)
        endif()
        list(APPEND release_copy_commands
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${target}>" "${release_bin_dir}"
            COMMAND ${CMAKE_COMMAND}
                "-DTARGET_PATH=$<TARGET_FILE:${target}>"
                "-DTARGET_DIR=${release_bin_dir}"
                "-DTARGET_KIND=${target_kind}"
                -P "${CMAKE_CURRENT_BINARY_DIR}/copy_runtime_dependencies.cmake"
        )
    endforeach()
    if(release_targets)
        add_custom_target(release
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${release_bin_dir}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${release_bin_dir}"
            ${release_copy_commands}
            COMMENT "Staging release binaries in ${release_bin_dir}"
            VERBATIM
        )
        add_dependencies(release ${release_targets})
    endif()
endfunction()
