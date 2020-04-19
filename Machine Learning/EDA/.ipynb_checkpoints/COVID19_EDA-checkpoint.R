library(dplyr)

confirmed_cases_url <- "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
covid_confirmed <- read.csv(confirmed_cases_url,header=TRUE, check.names = FALSE)

recovered_cases_url <- 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
covid_recovered <- read.csv(recovered_cases_url,header=TRUE, check.names = FALSE)

deceased_cases_url <- 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
covid_deceased <- read.csv(deceased_cases_url,header=TRUE, check.names = FALSE)

#utils::View(covid_confirmed)
?
                    spread()

head(covid_confirmed)