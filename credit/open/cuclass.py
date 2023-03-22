class LinearBlock():
    def init:
    
       linear
       relu
       dropout

    def forawrd:
       return


class Model:
    def init:
     self.linear= nn.sequential(
             LinearBlock,
              LinearBlock,
             LinearBlock,
             LinearBlock,
             LinearBlock,
          )

    def forawrd(x):
        return self.linear(x)