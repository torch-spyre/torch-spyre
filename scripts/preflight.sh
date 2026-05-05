#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="oc_install.log"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

<<<<<<< HEAD
OC_BIN="/usr/local/bin/oc"

log "Starting OpenShift CLI check"

# Check if oc exists AND is valid
if [ -x "$OC_BIN" ]; then
    log "oc binary found, validating..."

    if "$OC_BIN" version --client >/dev/null 2>&1; then
        log "oc is valid"
        "$OC_BIN" version --client | tee -a "$LOG_FILE"
        exit 0
    else
        log "Invalid oc binary detected. Removing..."
        sudo rm -f "$OC_BIN"
    fi
fi

log "Installing fresh oc binary..."
=======
log "Starting OpenShift CLI check"

if [ -x /usr/local/bin/oc ]; then
    log "oc already installed"
    /usr/local/bin/oc version --client | tee -a "$LOG_FILE"
    exit 0
fi

log "oc not found. Installing..."
>>>>>>> main

TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

log "Downloading oc client"
<<<<<<< HEAD
curl -LO https://mirror.openshift.com/pub/openshift-v4/clients/oc/latest/linux/oc.tar.gz

log "Extracting archive"
tar -xzf oc.tar.gz

log "Installing binary"
chmod +x oc
sudo mv oc "$OC_BIN"

log "Validating installation"
"$OC_BIN" version --client | tee -a "$LOG_FILE"

log "Installation completed"
=======
curl -LO https://mirror.openshift.com/pub/openshift-v4/clients/ocp/latest/openshift-client-linux.tar.gz

log "Extracting archive"
tar -xvf openshift-client-linux.tar.gz

log "Installing binary"
sudo mv oc /usr/local/bin/

log "Validating installation"
/usr/local/bin/oc version --client | tee -a "$LOG_FILE"

log "Installation completed"
>>>>>>> main
