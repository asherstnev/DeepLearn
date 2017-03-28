library(data.table)
library(RnavGraphImageData)
library(EBImage)
library(mxnet)

################################################################################################
# data sources
the.faces <- fread("../olivetti_photos.tsv")
the.targets <- fread("../olivetti_labels.tsv")
the.lebels <- the.targets[,1][[1]]

the.image <- as.matrix()

################################################################################################
# preprocessing: shrinking photos
shrink.photos <- function (d, orig.size = 64, new.size = 64, verbose=0) {
  orig.x.size <- orig.size
  orig.y.size <- orig.x.size
  
  x.size <- new.size
  y.size <- x.size
  
  # resize and set to greyscale
  resized.faces <- data.frame()
  for(n.photo in 1:nrow(d)) {
    the.img.vec <- as.numeric(d[n.photo,])
    # reshape as a (orig.x.size, orig.y.size) image
    the.img <- EBImage::Image(the.img.vec, dim=c(orig.x.size, orig.y.size), colormode = "Grayscale")
    # resize image to from (orig.x.size, orig.y.size) to (x.size, y.size)
    resized.img <- resize(the.img, w = x.size, h = y.size)
    # flatten to vector and combine with labels
    the.vec <- c(the.lebels[n.photo], as.vector(t(resized.img@.Data)))
    resized.faces <- rbind(resized.faces, the.vec)
    if (0 < verbose)
      print(paste0("Done photo ", n.photo))
  }
  rm(resized.img)
  
  # Set names. The first columns are the labels, the other columns are the pixels.
  colnames(resized.faces) <- c("target", paste0("pixel", c(1:(x.size * y.size))))
  resized.faces
}

new.faces <- shrink.photos(the.faces, 64, 32, verbose = 10)

# show new photes
show.face <- function(d, n.size = 64, num = 1) {
  the.face <- new.faces[num,]
  the.face <- the.face[2:length(the.face)]
  if (n.size^2 == length(the.face)) {
    the.face <- as.matrix(the.face)
    dim(the.face) <- c(n.size, n.size)
    image(the.face)
  } else {
    print(paste0("Wrong photo size = ", n.size))
  }
}
show.face(the.faces, n.size = 64, num = 127)

################################################################################################
# split into train and test sets
if (1) {
  set.seed(1234)
  train.frac <- 0.8
  n.train <- round(nrow(new.faces) * train.frac, 0)
  resized.faces <- new.faces[sample(1:400),]
  train.data <- new.faces[1:n.train, ]
  test.data <- new.faces[(n.train + 1):400, ]
}

################################################################################################
# training and testing CNN 1
if (1) {
  # prepate train and test datasets
  train.input <- t(train.data[, -1])
  dim(train.input) <- c(x.size, y.size, 1, ncol(train.input))

  test.input <- t(test.data[, -1])
  dim(test.input) <- c(x.size, y.size, 1, ncol(test.input))
  
  # configure the symbolic model
  data <- mx.symbol.Variable('data')
  # 1st convolutional layer
  conv.1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
  tanh.1 <- mx.symbol.Activation(data = conv.1, act_type = "tanh")
  pool.1 <- mx.symbol.Pooling(data = tanh.1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 2nd convolutional layer
  conv.2 <- mx.symbol.Convolution(data = pool.1, kernel = c(5, 5), num_filter = 50)
  tanh.2 <- mx.symbol.Activation(data = conv.2, act_type = "tanh")
  pool.2 <- mx.symbol.Pooling(data=tanh.2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
  # 1st fully connected layer
  flatten <- mx.symbol.Flatten(data = pool.2)
  fc.1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
  tanh.3 <- mx.symbol.Activation(data = fc.1, act_type = "tanh")
  # 2nd fully connected layer
  fc.2 <- mx.symbol.FullyConnected(data = tanh.3, num_hidden = 80)
  # Output. Softmax output since we'd like to get some probabilities.
  NN.graph <- mx.symbol.SoftmaxOutput(data = fc.2)
  
  # Train the model
  mx.set.seed(1234)
  cpu.devices <- list(mx.cpu(0), mx.cpu(1), mx.cpu(3), mx.cpu(4))
  # momentum - parameter of SGD (statistical gradient discent)
  system.time(
    the.model.1 <- mx.model.FeedForward.create(NN.graph
                                               , X = train.input, y = train.data$target
                                               , ctx = cpu.devices
                                               , num.round = 256
                                               , array.batch.size = 40
                                               , optimizer = "sgd"
                                               , learning.rate = 0.01
                                               , momentum = 0.9
                                               , eval.metric = mx.metric.accuracy
                                               , epoch.end.callback = mx.callback.log.train.metric(1)
                                               , verbose = T)
  )
  # testing the model
  predicted.mx <- predict(the.model.1, test.input)
  # Assign labels
  predicted.labels <- max.col(t(predicted.mx)) - 1
  # Get accuracy
  the.acc <- length(predicted.labels[test.data$target == predicted.labels])/length(predicted.labels)
  
  
  
  # visualize the model
  graph.viz(the.model$symbol)
  
}

################################################################################################
# training and testing CNN 2
if (1) {
  # prepate train and test datasets
  train.input <- t(train.data[, -1])
  dim(train.input) <- c(x.size, y.size, 1, ncol(train.input))
  
  test.input <- t(test.data[, -1])
  dim(test.input) <- c(x.size, y.size, 1, ncol(test.input))
  
  # configure the symbolic model
  if (1) {
    data <- mx.symbol.Variable('data')
    # 1st convolutional layer
    conv.1 <- mx.symbol.Convolution(data = data, kernel = c(5, 5), num_filter = 20)
    tanh.1 <- mx.symbol.Activation(data = conv.1, act_type = "tanh")
    pool.1 <- mx.symbol.Pooling(data = tanh.1, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
    # 2nd convolutional layer
    conv.2 <- mx.symbol.Convolution(data = pool.1, kernel = c(5, 5), num_filter = 50)
    tanh.2 <- mx.symbol.Activation(data = conv.2, act_type = "tanh")
    pool.2 <- mx.symbol.Pooling(data=tanh.2, pool_type = "max", kernel = c(2, 2), stride = c(2, 2))
    # 1st fully connected layer
    flatten <- mx.symbol.Flatten(data = pool.2)
    fc.1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 500)
    tanh.3 <- mx.symbol.Activation(data = fc.1, act_type = "tanh")
    # 2nd fully connected layer
    fc.2 <- mx.symbol.FullyConnected(data = tanh.3, num_hidden = 80)
    # Softmax output to get probabilities.
    NN.model.2 <- mx.symbol.SoftmaxOutput(data = fc.2)
  }
  
  # Train the model
  mx.set.seed(1234)
  cpu.devices <- list(mx.cpu(0), mx.cpu(1), mx.cpu(3), mx.cpu(4))
  system.time(
  the.model.2 <- mx.model.FeedForward.create(NN.model.2
                                           , X = train.input, y = train.data$target
                                           , ctx = cpu.devices
                                           , num.round = 256
                                           , array.batch.size = 40
                                           , learning.rate = 0.01
                                           , momentum = 0.9
                                           , eval.metric = mx.metric.accuracy
                                           #, epoch.end.callback = mx.callback.log.train.metric(100)
                                           , verbose = F)
  )
  # testing the model
  predicted.mx <- predict(the.model.2, test.input)
  # Assign labels
  predicted.labels <- max.col(t(predicted.mx)) - 1
  # Get accuracy
  the.acc <- length(predicted.labels[test.data$target == predicted.labels])/length(predicted.labels)
  
  # visualize the model
  graph.viz(the.model$symbol)
  
}


