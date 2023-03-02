import random
import numpy as np

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sklearn.metrics.pairwise import cosine_similarity

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
        # A dictionary to store vector ids and their corresponding index in the vectors array
        self.vector_ids = {}
        self.vectors = {}  # A dictionary to store vectors, grouped by their cosine similarity

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
                    FieldSchema(name='feature', dtype=DataType.FLOAT_VECTOR, dim=2)
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
        ids = self._generate_unique_ids(len(vectors))
        self.collection.insert(entities, ids)
        # Add vectors to appropriate groups based on cosine similarity
        for vector, id_ in zip(vectors, ids):
            self._add_vector_to_group(vector, id_)


    def _generate_unique_ids(self, num_ids):
        """
        Generates a list of unique IDs.

        Parameters
        ----------
        num_ids : int
            The number of IDs to generate.

        Returns
        -------
        List[int]
            A list of unique IDs.
        """
        existing_ids = set(self.collection.list_id())
        unique_ids = []
        while len(unique_ids) < num_ids:
            new_id = random.randint(0, 10000000)
            if new_id not in existing_ids:
                unique_ids.append(new_id)
                existing_ids.add(new_id)
        return unique_ids

    
    def _add_vector_to_group(self, vector, id_):
        """
        Adds a vector to the appropriate group based on cosine similarity.

        Parameters
        ----------
        vector : np.ndarray
            The vector represented as a numpy array.
        id_ : int
            The ID of the vector.
        """
        # Compute cosine similarity with all existing vectors
        cosine_similarities = []
        for group in self.vectors.values():
            for v in group:
                cosine_similarities.append(cosine_similarity([vector], [v])[0][0])
        # Find the group with the highest cosine similarity
        max_similarity = max(cosine_similarities, default=0)
        for similarity, group in self.vectors.items():
            if max_similarity == similarity:
                group.append(vector)
                break
            else:
                self.vectors[max_similarity] = [vector]
    
    def get_similar_vectors(self, vector, threshold=0.8):
        """
        Retrieves a list of vectors from the collection that are similar to the given vector based on cosine similarity.

        Parameters
        ----------
        vector : np.ndarray
            The vector to search for, represented as a numpy array.
        threshold : float, optional
            The minimum cosine similarity required for a vector to be considered similar. Defaults to 0.8.

        Returns
        -------
        List[np.ndarray]
            A list of vectors that are similar to the given vector.
        """
        similar_vectors = []
        for group in self.vectors.values():
            for v in group:
                cosine_sim = cosine_similarity([vector], [v])[0][0]
                if cosine_sim >= threshold:
                    similar_vectors.append(v)
        return similar_vectors

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
