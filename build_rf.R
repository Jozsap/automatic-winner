options(timeout=600)

suppressPackageStartupMessages({
  library(data.table)
  library(arrow)
  library(httr)
})

setwd("RFundamentals")
source("R/constituent_master.R")
source("R/sector_classifier.R")
source("R/fundamental_fetcher.R")
source("R/indicator_compute.R")
source("R/pit_assembler.R")
source("R/pipeline_runner.R")
source("R/timeseries_builder.R")
source("R/ttm_eps.R")

# Transport-only hardening for GitHub-hosted CI. The V9 strategy rules are
# untouched. Reuse the workflow-provided EDGAR_UA rather than embedding any
# additional contact data in this file.
.EDGAR_UA <- Sys.getenv("EDGAR_UA", .EDGAR_UA)
.EDGAR_RATE_SEC <- 0.20
.edgar_contact <- {
  m <- regmatches(.EDGAR_UA, regexpr("[^[:space:]()]+@[^[:space:]()]+", .EDGAR_UA))
  if (length(m) && nzchar(m)) m else ""
}

.edgar_fetch <- function(url, retries = 5L, timeout_s = 45) {
  last_status <- NA_integer_
  last_body <- ""
  for (attempt in seq_len(retries)) {
    hdrs <- c(
      `User-Agent` = .EDGAR_UA,
      Accept = "application/json,text/plain,*/*",
      `Accept-Encoding` = "gzip, deflate"
    )
    if (nzchar(.edgar_contact)) hdrs <- c(hdrs, From = .edgar_contact)

    resp <- tryCatch(
      httr::GET(url, httr::add_headers(.headers = hdrs), httr::timeout(timeout_s)),
      error = function(e) {
        message(sprintf("EDGAR transport error attempt %d/%d: %s", attempt, retries, e$message))
        NULL
      }
    )

    if (!is.null(resp)) {
      last_status <- httr::status_code(resp)
      if (last_status == 200L) {
        Sys.sleep(.EDGAR_RATE_SEC)
        return(httr::content(resp, as = "text", encoding = "UTF-8"))
      }
      last_body <- tryCatch(
        substr(httr::content(resp, as = "text", encoding = "UTF-8"), 1, 240),
        error = function(e) ""
      )
      message(sprintf("EDGAR HTTP %s attempt %d/%d", last_status, attempt, retries))
      if (nzchar(last_body)) message(sprintf("EDGAR body: %s", gsub("[\r\n]+", " ", last_body)))
    }

    if (attempt < retries) Sys.sleep(min(2^attempt, 16))
  }
  warning(sprintf("EDGAR request failed after %d attempts; last HTTP=%s; url=%s", retries, last_status, url), call. = FALSE)
  NULL
}

dir.create("cache/lookups", recursive = TRUE, showWarnings = FALSE)

message("SEC EDGAR smoke test: AAPL companyfacts")
smoke <- fetch_companyfacts("0000320193")
if (is.null(smoke) || is.null(smoke$entityName)) {
  stop("SEC EDGAR smoke test FAILED; aborting before full V9 data build")
}
message(sprintf("SEC EDGAR smoke test PASS: %s", smoke$entityName))

build_constituent_master()
build_sector_industry()
build_fundamentals()

fund_files <- list.files("cache/fundamentals", pattern = "\\.parquet$", full.names = TRUE)
message(sprintf("Fundamental cache coverage: %d parquet files", length(fund_files)))
if (length(fund_files) < 300L) {
  stop(sprintf("Fundamental coverage gate FAILED: only %d files (<300)", length(fund_files)))
}

build_timeseries(start_date = "2014-01-02", end_date = Sys.Date())

daily_files <- list.files("cache/timeseries", pattern = "_daily\\.parquet$", full.names = TRUE)
fund_ts_files <- list.files("cache/timeseries", pattern = "_fund\\.parquet$", full.names = TRUE)
message(sprintf("Timeseries coverage: daily=%d fund=%d", length(daily_files), length(fund_ts_files)))
if (length(daily_files) < 300L || length(fund_ts_files) < 300L) {
  stop(sprintf("Timeseries coverage gate FAILED: daily=%d fund=%d", length(daily_files), length(fund_ts_files)))
}

message("RFundamentals build PASS: coverage gates satisfied")
