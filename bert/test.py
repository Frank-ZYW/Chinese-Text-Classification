from train import evaluate


def test(config, model, test_data):
    """
    test function
    """
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_data, test=True)
    msg = '\nTest Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
