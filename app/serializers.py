from rest_framework import serializers

from .models import Recipe

"""Serializers for the recipes app.

This module defines serializers used by the API endpoints to serialize and
deserialize Recipe instances.
"""


class RecipeSerializer(serializers.ModelSerializer):
    """Serializer for the Recipe model.

    Exposes the following fields:
    - id: auto-generated primary key
    - title: short name/title of the recipe
    - description: optional detailed instructions
    - time_minutes: estimated preparation time in minutes
    - price: estimated cost to prepare the recipe

    The ModelSerializer base class will automatically generate serializer
    fields that correspond to the model's fields.
    """

    class Meta:
        model = Recipe
        # Fields included in API representations and (de)serialization.
        fields = ["id", "title", "description", "time_minutes", "price"]
        # Note: you can add `read_only_fields` or `extra_kwargs` here to
        # customize field behavior (e.g., making 'id' read-only).
