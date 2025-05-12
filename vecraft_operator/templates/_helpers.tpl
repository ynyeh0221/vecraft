{{- define "vecraft-operator.fullname" -}}
{{- printf "%s-%s" .Release.Name "operator" | trunc 63 }}
{{- end -}}