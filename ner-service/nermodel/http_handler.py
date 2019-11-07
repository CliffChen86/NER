from nermodel.model_wrapper import ModelWrapper
from nermodel.globals import RespType
import msgpack
import aiohttp.web

__all__ = ['HTTPHandler']


class HTTPHandler:
    def __init__(self,
                model: ModelWrapper,
                default_response_type: RespType = RespType.JSON):
        self._model = model
        self._default_response_type = default_response_type
    
    async def post(self, req):
        request = await req.json()
        query = request['q']
        print(query)
        ner = await self._model.predict(query)
        rsp_type = request.get("response_type", None)

        if isinstance(rsp_type, str):
            rsp_type = RespType[rsp_type.upper()]
        else:
            rsp_type = self._default_response_type
        
        if rsp_type == RespType.JSON:
            return aiohttp.web.json_response(ner)
        elif rsp_type == RespType.MSGPACK:
            rsp = aiohttp.web.Response(
                headers={'Content-Type': 'application/x-msgpack'},
                status=200,
                body=msgpack.packb(ner),
                reason='OK')
            return rsp