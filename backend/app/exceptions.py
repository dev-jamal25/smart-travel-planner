class AppError(Exception):
    """"""


class AuthError(AppError):
    """"""


class UserNotFoundError(AppError):
    """"""


class ToolFailureError(AppError):
    """"""


class WebhookDeliveryError(AppError):
    """"""
