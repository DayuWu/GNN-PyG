# https://github.com/EdisonLeeeee/GraphGallery

from graphgallery.gallery.nodeclas import GCN
trainer = GCN()
trainer.setup_graph(graph)
trainer.build()
history = trainer.fit(train_nodes, val_nodes)
results = trainer.evaluate(test_nodes)
print(f'Test loss {results.loss:.5}, Test accuracy {results.accuracy:.2%}')