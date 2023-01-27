variable "TAG" {
  default = "v0.5.2"
}

target "ark-analysis" {
  dockerfile = "Dockerfile"
  tags       = ["angelolab/ark-analysis:${TAG}"]
  platforms  = ["linux/arm64/v8", "linux/amd64"]
}

group "build" {
  targets = ["ark-analysis"]
}