const dataset = require("easy-mnist").makeData(60000, 10000)
const dannjs = require("dannjs")
const fs = require("fs")
const Dann = dannjs.dann


let bigboi = new Dann(784, 10)
bigboi.addHiddenLayer(128, "leakyReLU")
bigboi.addHiddenLayer(64, "leakyReLU")

bigboi.makeWeights()
bigboi.lr = 0.0001
bigboi.log()
for (let e = 0; e < 10; e++){
  for (data of dataset.traindata){
    bigboi.train(data.image, data.label)
  }
console.log("completed " + (e+1))
}


let bigboi_json = bigboi.toJSON()
fs.writeFile("models/nn.json", JSON.stringify(bigboi_json) ,function() {
console.log("Finish")
})
