import random
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType


class VectorDatabase:
    """
    A class that handles interactions with a Milvus vector database.

    Parameters
    ----------
    collection_name : str
        The name of the collection to store vectors in.
    host : str, optional
        The hostname or IP address of the Milvus server. Defaults to '127.0.0.1'.
    port : str, optional
        The port number of the Milvus server. Defaults to '19530'.
    """

    def __init__(self, collection_name, host='127.0.0.1', port='19530'):
        """
        Initializes the VectorDatabase object and connects to the Milvus server.

        Parameters
        ----------
        collection_name : str
            The name of the collection to store vectors in.
        host : str, optional
            The hostname or IP address of the Milvus server. Defaults to '127.0.0.1'.
        port : str, optional
            The port number of the Milvus server. Defaults to '19530'.
        """
        self.collection_name = collection_name
        self.client = connections.connect(host, port)
        self.collection = self._get_collection()

    def _get_collection(self):
        """
        Gets or creates the collection with the specified name.

        Returns
        -------
        Collection
            The collection object.
        """
        if self.collection_name in self.client.list_collections():
            return Collection(self.collection_name)
        else:
            collection_schema = CollectionSchema(
                self.collection_name,
                fields=[
                    FieldSchema(name='feature', dtype=DataType.FLOAT_VECTOR, dim=512)
                ],
                description="Reduced feature vectors for image data"
            )
            return Collection.create(collection_schema)

    def insert_vectors(self, vectors):
        """
        Inserts a list of feature vectors into the collection.

        Parameters
        ----------
        vectors : List[np.ndarray]
            A list of feature vectors, each represented as a numpy array.
        """
        entities = [{"feature": vector} for vector in vectors]
        ids = [random.randint(0, 10000000) for _ in range(len(vectors))]
        self.collection.insert(entities, ids)

    def get_vector_by_id(self, id_):
        """
        Retrieves a vector from the collection by ID.

        Parameters
        ----------
        id_ : int
            The ID of the vector to retrieve.

        Returns
        -------
        np.ndarray
            The vector represented as a numpy array.
        """
        return self.collection[id_]

    def delete_all_vectors(self):
        """
        Deletes all vectors from the collection and recreates it with the same schema.
        """
        self.collection.drop()
        self._get_collection()

    def get_collection_info(self):
        """
        Gets basic information about the collection.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - num_entities: The number of entities in the collection.
            - dimension: The dimensionality of the vectors in the collection.
            - description: The description of the collection.
        """
        collection_info = self.collection.get_info()
        num_entities = collection_info['num_entities']
        dimension = collection_info['schema']['feature']['params']['dim']
        description = collection_info['description']
        return {'num_entities': num_entities, 'dimension': dimension, 'description': description}
