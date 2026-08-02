#evaluate.py-auto-generated
def evaluate_model(model,val_ds):
    loss,acc = model.evaluate(val_ds)
    print(f"validation accuracy is : {acc*100:.2f}%")