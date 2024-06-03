from unpickle import unpickle_f

from Printing_images import printer



def main():
    files = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
    dicts, xs, ys, = unpickle_f(files)
    print(xs)
    #printer()




if "__name__" == main():
    main()


