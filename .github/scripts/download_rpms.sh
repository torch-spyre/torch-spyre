#!/usr/bin/env bash

function check_env_var() {
    local var_name="$1"
    local var_value="${!var_name}"
    if [[ -z "${var_value}" ]] ; then
        echo "please set the environment variable: $var_name"
        exit 1
    fi
}

function check_command_exists() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "${command_name}: command not found"
        exit 1
    fi
}

if [[ -f ".env" ]] ; then
    echo "found a .env file"
    source .env
fi

if [[ -z "$RPM_ARCH" ]] ; then
    echo "RPM_ARCH is not set, detecting..."
    RPM_ARCH="$(arch)"
    if [[ "$RPM_ARCH" == 'arm64' ]] ; then
        echo 'detected arch arm64, defaulting to x86_64'
        RPM_ARCH="x86_64"
    fi
    echo "RPM_ARCH set to $RPM_ARCH"
fi

check_command_exists dnf

check_env_var ARTIFACTORY_USER
check_env_var ARTIFACTORY_API_KEY

RPM_NAMES_TXT="${RPM_NAMES_TXT:-rpms.txt}"

if [[ -z "$RPM_NAMES" ]] ; then
    echo 'RPM_NAMES is empty, looking for RPM_NAMES_TXT'
    check_env_var RPM_NAMES_TXT
    if [[ ! -f "$RPM_NAMES_TXT" ]] ; then
        echo "The RPMs file is missing: ${RPM_NAMES_TXT}"
        exit 1
    fi
    RPM_NAMES="$(tr '\n' ' ' < "$RPM_NAMES_TXT")"
fi

check_env_var RPM_NAMES

if [[ -z "$ARTIFACTORY_LOCATION" ]] ; then
    echo 'ARTIFACTORY_LOCATION is empty, looking for ARTIFACTORY_BASE_URL and ARTIFACTORY_RPM_PATH'
    check_env_var ARTIFACTORY_BASE_URL
    check_env_var ARTIFACTORY_RPM_PATH
    ARTIFACTORY_LOCATION="${ARTIFACTORY_BASE_URL}/${ARTIFACTORY_RPM_PATH}"
fi

RPMS_DOWNLOAD_DIR="${RPMS_DOWNLOAD_DIR:-rpms}"

DNF_REPO_CONFIG_DIR='dnf_repo_config'
DNF_REPO_NAME='artifactory'

mkdir -p "$DNF_REPO_CONFIG_DIR" || exit 1

cat <<EOF > "${DNF_REPO_CONFIG_DIR}/artifactory.repo"
[${DNF_REPO_NAME}]
name=artifactory
baseurl=${ARTIFACTORY_LOCATION}
username=${ARTIFACTORY_USER}
password=${ARTIFACTORY_API_KEY}
enabled=1
gpgcheck=0
sslverify=1
EOF

echo "downloading rpm(s) '${RPM_NAMES}' for arch '${RPM_ARCH}' from Artifactory location '${ARTIFACTORY_LOCATION}' to directory '${RPMS_DOWNLOAD_DIR}'..."

# Check if the current user ID is NOT 0 (not root)
DNF_CLI="dnf"
if [[ "$EUID" -ne 0 ]]; then
    DNF_CLI="sudo dnf"
fi

echo $DNF_CLI download \
    --setopt="reposdir=${DNF_REPO_CONFIG_DIR}" \
    --repo="$DNF_REPO_NAME" \
    --arch="$RPM_ARCH" \
    --destdir="$RPMS_DOWNLOAD_DIR" \
    $RPM_NAMES || exit 1

$DNF_CLI download \
    --setopt="reposdir=${DNF_REPO_CONFIG_DIR}" \
    --repo="$DNF_REPO_NAME" \
    --arch="$RPM_ARCH" \
    --destdir="$RPMS_DOWNLOAD_DIR" \
    $RPM_NAMES || exit 1

echo 'done'
