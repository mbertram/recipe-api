from django.db import models
from django.conf import settings

"""Models for the recipes app.

This module defines the Recipe model used throughout the API.
"""


class Recipe(models.Model):
    """Model representing a cooking recipe.

    Fields
    ------
    title : CharField
        Short title/name of the recipe (max 200 chars).
    description : TextField
        Optional detailed description or instructions for the recipe.
    time_minutes : IntegerField
        Estimated time to prepare the recipe, in minutes.
    price : DecimalField
        Estimated cost to prepare the recipe. Stored as a decimal with
        two decimal places (max 5 digits, 2 decimals).
    """

    # Short name/title for the recipe.
    title = models.CharField(max_length=200, help_text="Short name/title for the recipe.")
    # Optional full description or step-by-step instructions.
    description = models.TextField(blank=True, help_text="Optional full description or instructions.")
    # Estimated preparation time in minutes.
    time_minutes = models.IntegerField(help_text="Estimated preparation time in minutes.")
    # Estimated price/cost to prepare the recipe.
    price = models.DecimalField(max_digits=5, decimal_places=2, help_text="Estimated price in local currency.")

    def __str__(self):
        """Return a human-readable representation of the recipe.

        We return the recipe title which is suitable for admin displays
        and logs.
        """
        return self.title


class RecipeRating(models.Model):
    recipe = models.ForeignKey(
        Recipe, on_delete=models.CASCADE, related_name="ratings"
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )
    stars = models.PositiveSmallIntegerField(
        choices=[(i, f"{i} star{'s' if i>1 else ''}") for i in range(1, 6)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("recipe", "user")
        ordering = ["-created_at"]