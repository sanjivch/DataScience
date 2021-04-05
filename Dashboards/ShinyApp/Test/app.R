library(shiny)
library(ggplot2)

animals <- c("man", "ape")
# This the UI
# define your UI elements here
ui <- fluidPage(
    textInput("name", "What's your name?"),
    numericInput("age", "How old are you?", 1),
    selectInput("dataset", label = "Dataset", choices = ls("package:datasets")),
    selectInput("dataset2", label = "Choices", choices = c(1,2,3)),
    passwordInput("pass", label="Password"),
    radioButtons("rad", label = "Select animal", choiceNames = list(
        icon("angry"),
        icon("smile"),
        icon("sad-tear")),
        choiceValues = list("angry", "happy", "sad")),
    fileInput("upload", label = "Select file"),
    actionButton("click", "Click me!", class = "btn-danger"),
    textInput("tes1", label="xxx", placeholder="Your name"),
    verbatimTextOutput("summary"),
    tableOutput("name"),
    tableOutput("age"),
    sliderInput("x", label = "If x is", min=1, max=50, value=30),
    sliderInput("y", label = "If y is", min=20, max=25, value=20),
    sliderInput("test1", label="both sides",  min = 0, max=50, value = c(10,20)),
    textOutput("product"),
    tableOutput("plot")
)


server <- function(input, output, session) {
    

        
    dataset <- reactive({
        get(input$dataset, "package:datasets")
    })
    output$summary <- renderPrint({
        dataset()
    })
    
    output$name <- renderText({
        paste0("Hello ", input$name, "!")
    })
    
    output$age <- renderText({
        paste("Aged", input$age)
    })
    output$table <- renderTable({
        
        head(dataset())
    })
    
   
    output$product <- renderText({
        input$x * input$y
    })
    
    
}


shinyApp(ui, server)