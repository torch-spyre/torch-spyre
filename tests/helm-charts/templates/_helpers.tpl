{{/*
Expand the name of the chart.
*/}}
{{- define "spyre-dashboard.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
Uses the release name. Truncated to 63 chars (Kubernetes label limit).
*/}}
{{- define "spyre-dashboard.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to all resources.
*/}}
{{- define "spyre-dashboard.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | trunc 63 | trimSuffix "-" }}
{{ include "spyre-dashboard.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
deployer: {{ .Values.deployer }}
{{- end }}

{{/*
Selector labels — used by the Service to find the Pod.
*/}}
{{- define "spyre-dashboard.selectorLabels" -}}
app.kubernetes.io/name: {{ include "spyre-dashboard.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app: {{ include "spyre-dashboard.fullname" . }}
{{- end }}