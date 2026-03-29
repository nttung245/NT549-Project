# Bucket to store website
resource "google_storage_bucket" "website" {
  name     = "example-website-by-tung"
  location = "US"
}

# Make new object public
resource "google_storage_object_access_control" "public_rule" {
  object = google_storage_bucket_object.static_site_src.name
  bucket = google_storage_bucket.website.name
  role   = "READER"
  entity = "allUsers"
}

# Upload the html file to the Bucket
resource "google_storage_bucket_object" "static_site_src" {
  name   = "index.html"
  bucket = google_storage_bucket.website.name
  source = "../website/index.html"
}
