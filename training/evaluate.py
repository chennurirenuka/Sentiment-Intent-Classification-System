def _train_and_evaluate_model(model, X_train, y_train, X_test, y_test, label_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Ensure classification report always matches full label list
    label_ids = list(range(len(label_names)))

    report = classification_report(
        y_test,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=label_ids,
    ).tolist()

    return model, report, weighted_f1, matrix

