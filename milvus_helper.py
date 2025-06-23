from pymilvus import MilvusClient


class MilvusHelper:
    def __init__(self, uri, token):
        """
        Initializes the MilvusClient using credentials from environment variables.
        """
        try:
            self.client = MilvusClient(uri=uri, token=token)
            print(f"Successfully initialized MilvusClient with local DB: {uri}")
        except Exception as e:
            print(f"Failed to initialize MilvusClient: {e}")
            raise

    def get_vector_by_id(self, collection_name, item_id):
        """s
        Retrieves a vector from a collection by its ID.

        Args:
            collection_name (str): The name of the collection.
            item_id (int): The ID of the item to retrieve.

        Returns:
            list: The vector corresponding to the item_id, or None if not found.
        """
        if not self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return None
        
        results = self.client.query(
            collection_name=collection_name,
            filter=f"id == {item_id}",
            limit=1,
            output_fields=["vector", "id", 'category']
        )
        if results:
            res = results[0].get('vector'), results[0].get('category')
            return res
        else:
            print(f"No item found with id {item_id} in collection '{collection_name}'.")
            return None

    def get_similar_items(self, collection_name, vector, category,  top_k=5):
        """
        Searches for the most similar items to an input vector, filtered by category.

        Args:
            collection_name (str): The name of the collection.
            vector (list): The input vector for the similarity search.
            category (str): The category to filter by.
            top_k (int): The number of similar items to return.

        Returns:
            list: A list of the top-k most similar items' IDs.
        """
        if not self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return []
            
        
        results = self.client.search(
            collection_name=collection_name,
            data=[vector],
            limit=top_k,
            filter=f"category != '{category}'",
            output_fields=["id"]
        )
        
        matched_id = []
        for result in results:
            for x in result:
                matched_id.append(x['entity']['id'])
        return matched_id

    def add_vector(self, collection_name, item_id, vector, category):
        """
        Adds a vector to a collection with its ID and category.

        Args:
            collection_name (str): The name of the collection.
            item_id (int): The ID of the item.
            vector (list): The vector to add.
            category (str): The category of the item.
        
        Returns:
            The result of the insert operation.
        """
        if not self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist. Please create it first.")
            return None
        
        data = [
            {"id": item_id, "vector": vector, "category": category}
        ]
        
        res = self.client.insert(collection_name=collection_name, data=data)
        return res

    def delete_vector_by_id(self, collection_name, item_id):
        """
        Deletes a vector from a collection by its ID.

        Args:
            collection_name (str): The name of the collection.
            item_id (int): The ID of the item to delete.
            
        Returns:
            The result of the delete operation.
        """
        if not self.client.has_collection(collection_name):
            print(f"Collection '{collection_name}' does not exist.")
            return None
            
        res = self.client.delete(collection_name=collection_name, ids=[item_id])
        return res

