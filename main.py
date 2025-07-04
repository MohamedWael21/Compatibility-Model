from flask import Flask, request, jsonify
import os
from model import SiameseEfficientNet
from milvus_helper import MilvusHelper
import tempfile
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Initialize models and helpers
compatibility_model = SiameseEfficientNet()
compatibility_model.load_model("best_siamese_model.pth")
milvus_helper = MilvusHelper(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN")
)

COLLECTION_NAME = "compatibility_embedding"

@app.route('/api/v1/items/recommendations/<int:item_id>', methods=['GET'])
def get_recommendations(item_id):
    """
    Get recommendations for a specific item by ID.
    
    Args:
        item_id (int): The ID of the item to get recommendations for
        
    Query Parameters:
        top_k (int): Number of recommendations to return (default: 5)
        
    Returns:
        JSON response with recommended item IDs
    """
    try:
        # Get top_k from query parameters, default to 5
        top_k = request.args.get('top_k', 5, type=int)
        
        # Validate top_k
        if top_k <= 0:
            return jsonify({
                'error': 'top_k must be a positive integer'
            }), 400
        
        # Get the vector and category for the specified item
        result = milvus_helper.get_vector_by_id(COLLECTION_NAME, item_id)
        
        if not result:
            return jsonify({
                'error': f'Item with ID {item_id} not found'
            }), 404
        
        vector, category = result
        
        # Get similar items in the same category
        similar_items = milvus_helper.get_compatible_items(
            collection_name=COLLECTION_NAME,
            vector=vector,
            category=category,
            top_k=top_k
        )
        
        return jsonify({
            'recommendations': similar_items
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting recommendations: {str(e)}'
        }), 500

@app.route('/api/v1/items', methods=['POST'])
def add_item():
    """
    Add a new item to the database.
    
    Expected form data:
    - id: int
    - category: string
    - image: file upload
    
    Returns:
        JSON response with success status
    """
    try:
        # Check if all required fields are present
        if 'id' not in request.form or 'category' not in request.form or 'image' not in request.files:
            return jsonify({
                'error': 'Missing required fields: id, category, image'
            }), 400
        
        item_id = int(request.form['id'])
        category = request.form['category']
        image_file = request.files['image']
        
        # Validate image file
        if image_file.filename == '':
            return jsonify({
                'error': 'No image file selected'
            }), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image_file.save(temp_file.name)
            temp_file_path = temp_file.name
        
        try:
            # Get image vector using the similarity model
            image = Image.open(temp_file_path).convert('RGB')
            embedding = compatibility_model(image)
            embedding_list = embedding.detach().numpy().flatten() # type: ignore[attr-defined]
            # Add to Milvus database
            result = milvus_helper.add_vector(
                collection_name=COLLECTION_NAME,
                item_id=item_id,
                vector=embedding_list,
                category=category
            )
            
            if result is None:
                return jsonify({
                    'error': 'Failed to add item to database'
                }), 500
            
            return jsonify({
                'message': 'Item added successfully',
                'item_id': item_id,
                'vector_dimension': len(embedding_list)
            }), 201
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    except ValueError as e:
        print(e)
        return jsonify({
            'error': 'Invalid ID format. ID must be an integer.'
        }), 400
    except Exception as e:
        print(e)
        return jsonify({
            'error': f'Error adding item: {str(e)}'
        }), 500

@app.route('/api/v1/items/<int:item_id>', methods=['DELETE'])
def delete_item(item_id):
    """
    Delete an item from the database by ID.
    
    Args:
        item_id (int): The ID of the item to delete
        
    Returns:
        JSON response with success status
    """
    try:
        # Check if item exists
        result = milvus_helper.get_vector_by_id(COLLECTION_NAME, item_id)
        
        if not result:
            return jsonify({
                'error': f'Item with ID {item_id} not found'
            }), 404
        
        # Delete the item
        delete_result = milvus_helper.delete_vector_by_id(COLLECTION_NAME, item_id)
        
        if delete_result is None:
            return jsonify({
                'error': 'Failed to delete item from database'
            }), 500
        
        return jsonify({
            'message': f'Item with ID {item_id} deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error deleting item: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Compatibility Model API is running'
    })

if __name__ == '__main__':
    # Ensure the collection exists
    print("Starting Flask application...")
    app.run(host='0.0.0.0', port=8080)
