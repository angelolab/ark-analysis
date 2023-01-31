group "build" {
  targets = ["ark-analysis"]
}

target "ark-analysis" {
  dockerfile = "Dockerfile"
  tags       = ["angelolab/ark-analysis:latest"]
  platforms  = ["linux/arm64", "linux/amd64"]

}