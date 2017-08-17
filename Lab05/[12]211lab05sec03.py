from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from IPython.display import Image as im
from PIL import Image, ImageDraw

iris = load_iris();
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

# with open("iris.dot", 'w') as f:
#     f = tree.export_graphviz(clf, out_file=None)

# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")
# graph.write_png("iris.png")

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names,
                                filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
im(graph.create_png())
graph.write_png("iris.png")
graph.write_pdf("iris.pdf")
image = Image.open("iris.png")
image.show()

print clf.predict(iris.data[:1, :])
print clf.predict_proba(iris.data[:1, :])

