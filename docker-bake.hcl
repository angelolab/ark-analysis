variable "TAG" {
  default = "v0.4.1"
}

target "ark-analysis" {
  dockerfile = "Dockerfile"
  tags       = ["angelolab/ark-analysis:${TAG}"]
  platforms  = ["linux/arm64", "linux/amd64"]
}

group "build" {
  targets = ["ark-analysis"]
}
