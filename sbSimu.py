import math
import cairo


# in my structure points are particles with properties free to change by springs and colliders
# each spring is related to two points, manipulate them
# colliders provide methods to check points' position and move them sometimes
# softbodies are filled with particles connected to springs, and they have integrated move method (including damping, gravity, collider collision, spring, and particles' self collision)
# collider collision has the highest priority, it

class Vector2():
    def __init__(self, x, y):
        self.value = (x, y)
        self.x = self.value[0]
        self.y = self.value[1]

    def __repr__(self):
        return repr(self.value)

    def __add__(self, other):
        return Vector2(self.value[0] + other.value[0], self.value[1] + other.value[1])

    def __sub__(self, other):
        return Vector2(self.value[0] - other.value[0], self.value[1] - other.value[1])

    def __mul__(self, other):  # cross_product or magnify
        if isinstance(other, Vector2):
            return self.value[0] * other.value[1] - self.value[1] * other.value[
                0]  # https://blog.csdn.net/ld326/article/details/84689620
        elif isinstance(other, (int, float)):  # only post-multiply of float works
            return Vector2(self.value[0] * other, self.value[1] * other)
        else:
            raise TypeError(f'multiplication of <class \'Vector2\'> with {type(other)}')

    def dot(self, other):
        return self.value[0] * other.value[0] + self.value[1] * other.value[1]

    # 没有向量与向量的除法运算
    def __truediv__(self, other):
        return Vector2(self.value[0] / other, self.value[1] / other)

    def mag(self):  # return magnitude
        return math.hypot(self.value[0], self.value[1])

    def normalize(self):
        return self / self.mag()

    def projection(self, other):  # project self vector to the direction of other
        return self.dot(other.normalize())

    def perpen(self):  # the perpendicular vector
        return Vector2(-self.y, +self.x)

    @classmethod
    def multipleVectors(cls, *args):  # type in all the vectors' components in sequence, I will handle them for you
        l = len(args)
        if l % 2 == 1:
            raise Exception("Odd number of arguments given (%d)" % l)
        return [cls(args[i], args[i + 1]) for i in range(0, l - 1, 2)]


# class cartesianCairoContext(cairo.Context): #我也想继承，但是C语言的库，我怎么拿到我的__init
class cartesianCoord:  # store drawing config
    def __init__(self, originX=1920 / 2, originY=1080 / 2, px2nScale=200,
                 img_width=1920, img_height=1080, gridcolor=(0.6, 0.6, 0.6)):
        self.originX = originX  # x coord in picture of origin
        self.originY = originY
        self.img_width = img_width
        self.img_height = img_height
        self.px2nScale = px2nScale  # the scale of (pixel of number/number)
        self.gridcolor = gridcolor

    def drawCoordinate(self, imageSurface):
        context = cairo.Context(imageSurface)
        h_num = self.img_width // self.px2nScale + 1  # No. of lines in horizontal direction
        v_num = self.img_height // self.px2nScale + 1
        context.set_source_rgb(*self.gridcolor)
        for i in range(-int(h_num / 2), int(h_num / 2)):
            x = self.originX + i * self.px2nScale
            context.move_to(x, 0)
            context.line_to(x, self.img_height)
        for i in range(-int(v_num / 2), int(v_num / 2) + 1):
            y = self.originY + i * self.px2nScale
            context.move_to(0, y)
            context.line_to(self.img_width, y)
        context.stroke()


class Point():
    def __init__(self, mass, position, velocity, g=Vector2(0, -9.81)):
        self.mass = mass
        self.pos = position
        self.v = velocity
        self.f = Vector2(0, 0)
        self.g = g
        if not isinstance(velocity, Vector2): raise TypeError(
            f"Point should have velocity of Vector2, instead you provided {type(velocity)}")
        if not isinstance(position, Vector2): raise TypeError(
            f"Point should have position of Vector2, instead you provided {type(position)}")

    def __repr__(self):
        return repr(f"Point(m={self.mass},pos={self.pos},v={self.v},f={self.f},g={self.g})")

    def clearFroce(self):
        self.f = 0

    def changePos(self, vector):
        '''
        Changes point pos
        :param vector: <class 'Vector2'> direction to move
        :return: 0
        '''
        self.pos += vector

    def changeV(self, vChange):
        self.v += vChange

    def move(self, dt):  # if forces are all set already, use this to move (except gravity)
        acc = self.f / self.mass + self.g
        self.v += acc * dt
        self.pos += self.v * dt

    @classmethod
    def drawMultipleSelf(cls, cairoImageSurface, cartesianCoord, points, color=(1, 0, 0)):
        '''
        as its name suggests
        :param cartesianCoord: <class 'cartesianCoord'>
        :param points: list of <class 'Point'> objects
        :return: None
        '''
        context = cairo.Context(cairoImageSurface)
        for point in points:
            context.set_source_rgb(*color)
            context.arc(cartesianCoord.px2nScale * (point.pos.x) + cartesianCoord.originX,
                        cairoImageSurface.get_height() - (
                                    cartesianCoord.px2nScale * (point.pos.y) + cartesianCoord.originY),
                        0.05 * cartesianCoord.px2nScale, 0, 2 * math.pi)
            context.fill()


class Spring():
    def __init__(self, point1, point2, length, k):
        '''
        spring controls points within its subset
        :param point1: <class 'Point'>
        :param point2: <class 'Point'>
        :param point3: <class 'int'>
        :param k: <class 'int'> spring constant
        '''
        self.a = point1
        self.b = point2
        self.l = length  # original length
        self.k = k

    def gf(self):  # give forces between them, positive if pushing together
        dl = (self.a.pos - self.b.pos).mag() - self.l
        d = (self.a.pos - self.b.pos).normalize()
        print(d * self.k * dl)
        return d * self.k * dl

    def damping(self, constant=0):  # there isn't damping when constant is 0; # damping along the spring
        # return (self.a.pos-self.b.pos)*(self.a.pos - self.b.pos).normalize().dot((self.a.v - self.b.v)) * constant
        return Vector2(0,0)

def segmentsIntersect(p1, p2, p3, p4, ignore_dot_error=False):
    '''
    detect intersection between segment p1p2 and segment p3p4 in mathematical sense(including overlap)
    :param p1: <class 'Vector2'>, coordinate of a point
    :param p2: 
    :param p3: 
    :param p4:
    :param ignore_dot_error: When True and dot(s) is(are) present, check if dot is on line(or dots at same location)
    :return: True if intersect
    '''
    if p1.value == p2.value or p3.value == p4.value:
        if ignore_dot_error:
            if p1.value == p2.value:
                if p3.value != p4.value:
                    return pointOnSegment(p1, p3, p4)
                else:
                    return p1.value == p3.value
            else:
                return pointOnSegment(p3, p1, p2)
        else:
            raise Exception("At least one of the line is a dot")
    # if their rectangle boxes overlap(necessary for assertion in later steps)
    if p3.x < p4.x:
        minx = p3.x
        maxx = p4.x
    else:
        minx = p4.x
        maxx = p3.x
    if p3.y < p4.y:
        miny = p3.y
        maxy = p4.y
    else:
        miny = p4.y
        maxy = p3.y
    if (p1.x < minx and p2.x < minx) or (p1.x > maxx and p2.x > maxx):
        return False
    if (p1.y < miny and p2.y < miny) or (p1.y > maxy and p2.y > maxy):
        return False

    if (p2.x - p1.x) == 0 or (p4.x - p3.x) == 0:  # if vertical(only one line could because of rectangular detection
        return True
    else:
        # set up simultaneous equations, tangents of lines k1, k2, provided that lines are not vertical
        k1 = (p2.y - p1.y) / (p2.x - p1.x)
        k2 = (p4.y - p3.y) / (p4.x - p3.x)
        # detect if lines are parallel(including horizontal)
        if k1 == k2:  # y=kx+b, we find if b are equal, if they are it is true because of rectangle detection
            if (p1.y - k1 * p1.x) == (p3.y - k2 * p3.x):
                return True
            else:
                return False

        n = (p1.y - p3.y + k1 * (p3.x - p1.x)) / (k2 - k1)
        m = (p3.x - p1.x) + n
        if 0 <= abs(n) <= abs(p4.x - p3.x) and 0 <= abs(m) <= abs(p2.x - p1.x):
            return True
        else:
            return False


def segmentsIntersectExcluVert(p1, p2, p3, p4, ignore_dot_error=False):
    '''
    detect intersection between segment p1p2 and segment p3p4 in mathematical sense(including overlap)(Excluding vertices)
    :param p1: <class 'Vector2'>, coordinate of a point
    :param p2:
    :param p3:
    :param p4:
    :return: True if intersect
    '''
    if p1.value == p2.value or p3.value == p4.value:
        if ignore_dot_error:
            if p1.value == p2.value:
                if p3.value != p4.value:
                    return pointOnSegment(p1, p3, p4)
                else:
                    return p1.value == p3.value
            else:
                return pointOnSegment(p3, p1, p2)
        else:
            raise Exception("At least one of the line is a dot")
    # if their rectangle boxes overlap(necessary for assertion in later steps)
    if p3.x < p4.x:
        minx = p3.x
        maxx = p4.x
    else:
        minx = p4.x
        maxx = p3.x
    if p3.y < p4.y:
        miny = p3.y
        maxy = p4.y
    else:
        miny = p4.y
        maxy = p3.y
    if (p1.x <= minx and p2.x <= minx) or (p1.x >= maxx and p2.x >= maxx):
        return False
    if (p1.y <= miny and p2.y <= miny) or (p1.y >= maxy and p2.y >= maxy):
        return False

    if (p2.x - p1.x) == 0 or (p4.x - p3.x) == 0:  # if vertical(only one line could because of rectangular detection
        return True
    else:
        # set up simultaneous equations, tangents of lines k1, k2, provided that lines are not vertical
        k1 = (p2.y - p1.y) / (p2.x - p1.x)
        k2 = (p4.y - p3.y) / (p4.x - p3.x)
        # detect if lines are parallel(including horizontal)
        if k1 == k2:  # y=kx+b, we find if b are equal, if they are it is true because of rectangle detection
            if (p1.y - k1 * p1.x) == (p3.y - k2 * p3.x):
                return True
            else:
                return False

        n = (p1.y - p3.y + k1 * (p3.x - p1.x)) / (k2 - k1)
        m = (p3.x - p1.x) + n
        if 0 < abs(n) < abs(p4.x - p3.x) and 0 < abs(m) < abs(p2.x - p1.x):
            return True
        else:
            return False


def pointOnSegment(p1, p2, p3):
    '''
    if point p1 is on line p2p3
    :param p1: <class 'Vector2'>, the point
    :param p2: one end of the line
    :param p3: another end of the line
    :return: True/False
    '''
    return (p1 - p2) * (p1 - p3) == 0  # cross product


class Collider:  # stationary polygon for particles to collide elastically
    # each point is an Vector2() object
    def __init__(self, *points):
        self.vert = [*points]  # vertices
        self.sides = len(points)
        for i in self.vert:
            if not isinstance(i, Vector2):
                raise TypeError(f"Element {i} provided is not <class 'Vector2'>")
        side_intersect_det = self._self_intersect_det()
        if side_intersect_det:
            raise Exception(f"PolygonError: Intersecting Sides - {side_intersect_det[1]} to "
                            f"{side_intersect_det[0]} crosses {side_intersect_det[3]} to {side_intersect_det[2]}")

    def _self_intersect_det(self):  # detect if segments overlap
        for ip1 in range(-1, self.sides - 2):
            for ip3 in range(ip1 + 1, self.sides - 1):

                if segmentsIntersectExcluVert(self.vert[ip1], self.vert[ip1 + 1], self.vert[ip3], self.vert[ip3 + 1]):
                    return self.vert[ip1], self.vert[ip1 + 1], self.vert[ip3], self.vert[ip3 + 1]
        return False

    def withinPolygon(self, p1):  # raycasting # on the rim is not inside
        '''
        p1 is within the polygon. overlap of ray with side of polygon is counted as 0(because two vertices crossed)(vertices are counted)
        :param p1: <class 'Point'>, has a ray vertically upwards going through
        :return: True(within,counter is odd)/False(not,counter is even)
        '''
        span = len(self.vert)
        counter = 0
        for i in range(0, span - 1):
            p2, p3 = self.vert[i], self.vert[i + 1]
            if min(p2.x, p3.x) <= p1.pos.x <= max(p2.x, p3.x):
                if (p2.y + (p3.y - p2.y) / (p3.x - p2.x) * (p1.pos.x - p2.x)) >= p1.pos.y:
                    if not pointOnSegment(p1.pos, p2, p3):  # in case that the point is not inside but on the rim
                        counter += 1
        if counter % 2 == 1:
            return True
        else:
            return False

    def elasticCollision(self, p1):
        '''
        # Push the point p1 to the closest point on the polygon and return velocity after this
        # return value should be moved to, you should then apply Point.changePointPos() method
        # this only works when the point is within the polygon to a small depth. Otherwise it crashes
        :param p1: Point object
        :return: first Vector2 to push point outside the Colllider, second Vector2 to change the velocity of the point to
        '''
        minIndex = -1
        mindist = abs((p1.pos - self.vert[-1]) * (self.vert[-1] - self.vert[0]))
        for i in range(0, self.sides - 1):
            dist = abs((p1.pos - self.vert[i]) * (self.vert[i + 1] - self.vert[i]).normalize())  # scalar
            vectorToMove = (self.vert[i + 1] - self.vert[i]).perpen() * dist
            if not segmentsIntersect(p1.pos, p1.pos + vectorToMove, self.vert[i],
                                     self.vert[i + 1], ignore_dot_error=True):  # 垂足不在边上
                continue
            if dist < mindist:
                mindist = dist  # minimum distance
                minIndex = i
            elif dist == mindist:
                raise FutureWarning(
                    "Jesus Christ!!! I am hopeless. This should be only theoretically possible. Now we're done\nIndex = %d, Current Minimum dist = %f" % (i, mindist))
        # print(minIndex)
        return (self.vert[minIndex] - self.vert[minIndex + 1]).normalize().perpen() * mindist, self._elastColliV(p1.v, self.vert[minIndex], self.vert[minIndex + 1])

    def drawSelf(self, cairoImageSurface, cartesianCoord, color=(0, 0, 0)):
        ctx = cairo.Context(cairoImageSurface)
        ctx.set_source_rgb(*color)
        for i in range(-1, self.sides):
            ctx.line_to(self.vert[i].x * cartesianCoord.px2nScale + cartesianCoord.originX,
                        cairoImageSurface.get_height() - (
                                    self.vert[i].y * cartesianCoord.px2nScale + cartesianCoord.originY))
        ctx.stroke()

    def _elastColliV(self, v, p2, p3):
        '''
        calculate the final velocity after the ball has hit the collider's surface
        :param v: <class 'Vector2'> incident velocity of the particle
        :param p2: start of the collided edge
        :param p3: end of the collided edge
        :return: <class 'Vector2'> for velocity
        '''
        n = (p2 - p3).normalize().perpen()
        return n * abs(v.dot(n)) * -2 # 只能用点乘，否则垂直碰撞没法办


class SoftBody:
    def __init__(self,points,springs,g=Vector2(0,-9.81)):
        self.points = points # layer by layer, the points is an 2-D array (because we are in two dimension)
        self.springs = springs
        self.g = g
        for i in range(len(self.points)):
            for p in self.points[i]:
                p.g = g

    def drawSelf(self, cairoImageSurface, cartesianCoordinate, sprColor=(0.8, 0.8, 0.8), pointColor=(1, 0, 0)):
        context = cairo.Context(cairoImageSurface)
        context.set_source_rgb(*sprColor)
        for s in self.springs:
            context.move_to(s.a.pos.x * cartesianCoordinate.px2nScale + cartesianCoordinate.originX,
                            cairoImageSurface.get_height() - (s.a.pos.y * cartesianCoordinate.px2nScale + cartesianCoordinate.originY))
            context.line_to(s.b.pos.x * cartesianCoordinate.px2nScale + cartesianCoordinate.originX,
                            cairoImageSurface.get_height() - (s.b.pos.y * cartesianCoordinate.px2nScale + cartesianCoordinate.originY))
        context.stroke()
        for i in self.points:
            Point.drawMultipleSelf(cairoImageSurface, cartesianCoordinate, i, pointColor)
        context.stroke()

class SoftBodyRect(SoftBody):  # store information about a softbody
    def __init__(self, hN, vN, width, height, mass, pos, springK, g=Vector2(0, -9.81)):
        '''
        generate Point objects connected with Spring objects for a rectangular soft body
        :param hN: horizontal number of particles
        :param vN: vertical number of particles
        :param width: width relative to the cartesian coordinate
        :param height:
        :param mass: total mass
        :param pos: <class 'Vector2'> position of the bottom left particle
        :param springK: spring constant
        :param g: acceleration of free fall
        '''
        self.points = [[] for i in range(vN)]  # [[1,2,3],[4,5,6],[7,8,9],...]
        self.springs = []
        self.g = g
        unitMass = mass / hN / vN
        xIncre = width / (hN - 1)  # -1 is the number of separation
        yIncre = height / (vN - 1)
        diagIncre = math.hypot(xIncre, yIncre)
        for i in range(hN):
            for j in range(vN):
                self.points[j].append(Point(unitMass, pos + Vector2(i * xIncre, j * yIncre), Vector2(0, 0), self.g))
        for i in range(hN - 1):
            for j in range(vN - 1):
                self.springs.append(Spring(self.points[j][i], self.points[j][i + 1], length=xIncre, k=springK))
                self.springs.append(Spring(self.points[j][i], self.points[j + 1][i + 1], length=diagIncre, k=springK))
        for i in range(hN - 1):
            self.springs.append(Spring(self.points[vN - 1][i], self.points[vN - 1][i + 1], length=xIncre, k=springK))
        for j in range(1, vN):
            for i in range(hN - 1):
                self.springs.append(Spring(self.points[j][i], self.points[j - 1][i], length=yIncre, k=springK))
                self.springs.append(Spring(self.points[j][i], self.points[j - 1][i + 1], length=diagIncre, k=springK))
        for j in range(vN - 1):
            self.springs.append(Spring(self.points[j][hN - 1], self.points[j + 1][hN - 1], length=yIncre, k=springK))


class frameHandler:  # interact objects in scene (one softbody only)
    def __init__(self, dt, width, height, softbody, colliders, filename=None):
        self.dt = dt
        self.softboby = softbody
        self.colliders = colliders
        self.width = width
        self.height = height
        self.filename = filename

    def resolveFrames(self, cartesianCoordinate, maxFrameNumber):
        import os
        if self.filename==None:
            self.filename = os.path.basename(__file__)[:3]
        for i in range(maxFrameNumber):
            lastSoftbody = self.softboby
            imageSurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
            cartesianCoordinate.drawCoordinate(imageSurface)
            for s,s_new in zip(lastSoftbody.springs,self.softboby.springs):
                force = s.gf()
                dampingForce = s.damping()
                s_new.a.f -= force-dampingForce
                s_new.b.f += force-dampingForce
            for j in range(len(lastSoftbody.points)):
                for p,p_new in zip(lastSoftbody.points[j],self.softboby.points[j]):
                    for cld in self.colliders:
                        # if i//10==8:
                        #     print(' ---------------------------------------- ')
                        #     print(cld.withinPolygon(p))
                        #     print(' ---------------------------------------- ')
                        if cld.withinPolygon(p):
                            posChange, vChange = cld.elasticCollision(p)
                            p_new.changePos(posChange)  # collision of cld1 with p
                            p_new.changeV(vChange)
                    p.move(self.dt)
            for cld in self.colliders:
                cld.drawSelf(imageSurface, cartesianCoordinate)
            self.softboby.drawSelf(imageSurface,cartesianCoordinate)
            imageSurface.write_to_png(os.path.join(os.curdir, f"{self.filename}-{i}.png"))

def dbrf(sb, colid, dt, cartesianCoordinate, maxFrameNumber):
    import os
    filename = os.path.basename(__file__)
    for i in range(maxFrameNumber):
        lastSoftbody = sb
        imageSurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
        cartesianCoordinate.drawCoordinate(imageSurface)
        for s,s_new in zip(lastSoftbody.springs,sb.springs):
            force = s.gf()
            dampingForce = s.damping()
            s_new.a.f -= force-dampingForce
            s_new.b.f += force-dampingForce
        for j in range(len(lastSoftbody.points)):
            for p,p_new in zip(lastSoftbody.points[j],sb.points[j]):
                for cld in colid:
                    # if i//10==8:
                    #     print(' ---------------------------------------- ')
                    #     print(cld.withinPolygon(p))
                    #     print(' ---------------------------------------- ')
                    if cld.withinPolygon(p):
                        posChange, vChange = cld.elasticCollision(p)
                        p_new.changePos(posChange)  # collision of cld1 with p
                        p_new.changeV(vChange)
                p.move(dt)
        for cld in colid:
            cld.drawSelf(imageSurface, cartesianCoordinate)
        sb.drawSelf(imageSurface,cartesianCoordinate)
        imageSurface.write_to_png(os.path.join(os.curdir, f"{filename}-{i}.png"))

if __name__ == '__main__':
    print('# ----------initialising-canvas------------------------------ #')
    import os

    WIDTH, HEIGHT = 1920, 1080
    FILENAME = os.path.basename(__file__)
    ims = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    # # 狗儿Context没法继承，你叫我怎么办？
    cCoord = cartesianCoord(originX=WIDTH / 2, originY=HEIGHT / 2
                            , px2nScale=100, img_width=WIDTH, img_height=HEIGHT)
    # cCoord.drawCoordinate(ims)
    print('# ----------intersecting-segments----------------------------- #')
    # p1 = Vector2(-1, 0)
    # p2 = Vector2(-.5, 0)
    # p3 = Vector2(0, 0)
    # p4 = Vector2(5, 0)
    # print(segmentsIntersect(p1, p2, p3, p4))
    # print(segmentsIntersect(*Vector2.multipleVectors(-1,3, 4,4, 1,4, 3,3)))

    print('# -----------collider--and-\'within\'-detection----------------- #')
    cld1 = Collider(*Vector2.multipleVectors(0, 0, 2, 1, 3, 3, 1, 4, 2, 2, 0, 4, 1.5, 2, 0, 1, -1, 0))
    # cld2 = Collider(*Vector2.multipleVectors(0, 0, 2, 1, 3, 3, 1, 4, 2, 2, 0, 4, 1.5, 2, 1, 1))
    # # cld3 = Collider(*Vector2.multipleVectors(0,0,2,1,3,3,1,4,2,2,0,4,1.5,2,1,0.5)) # This raises an error
    # cld4 = Collider(*Vector2.multipleVectors(0, 0, 2, 1, 3, 3, 1, 4, 2, 2, 0, 4, 1.5, 2, 1, 0.75))
    # testPoint = Point(0,Vector2(1, 1),Vector2(0,0))
    # print(f"Point {testPoint} is within Collider1: {cld1.withinPolygon(testPoint)}")
    # print(f"Point {testPoint} is within Collider2: {cld2.withinPolygon(testPoint)}")
    # print(f"Point {testPoint} is within Collider3: {cld4.withinPolygon(testPoint)}")
    # # cld5 = Collider(*Vector2.multipleVectors(0,-1, 2,1, 3,3, 1,4, -1,3, 4,4, 1,5, -2,4, -1,0))
    # # testPoint = Vector2(1,1)
    # # print(f"Point {testPoint} is within Collider5: {cld5.withinPolygon(testPoint)}")
    # del testPoint
    print('# -----------pushout-of-point-\'within\'-polygon---------------- #')
    posVectors = Vector2.multipleVectors(1,0.6, 1.75,1, 2.1,3.25, -0.5,0.4,
                                         1,2.7, 2.5,2.1, 0,0.8, 0.5,1.3,
                                         -0.5,0.1,  1.5,3, 1.4,3, 1.5,3.5,
                                         1.5,2.25)
    points = []
    for i in range(len(posVectors)):
        points.append(Point(2.5*i+2.5,posVectors[i],Vector2(0,0)))
    Point.drawMultipleSelf(ims, cCoord, points,color=(0,0.8,0))
    # points that falls on vertices raises an error （such as point 2,1）
    # 0,0.8 is a very tricky but rare situation for 'within' polygon detection. I cannot solve this. See it for yourself.
    for p in points:
        if cld1.withinPolygon(p):
            posChange,vChange = cld1.elasticCollision(p)
            p.changePos(posChange) # collision of cld1 with p
            p.changeV(vChange)
    from pprint import pprint
    pprint(points)
    cld1.drawSelf(ims,cCoord)
    Point.drawMultipleSelf(ims,cCoord,points)

    print('# ---------------------Soft-body-rendering-------------------- #')
    # blc = Vector2(-1, 1)  # bottom left corner
    # sb = SoftBodyRect(hN=4, vN=6, width=2, height=3, mass=150, pos=blc, springK=2.37)
    # sb = SoftBodyRect(hN=2, vN=2, width=2, height=3, mass=150, pos=blc, springK=2.37)
    # sb.drawSelf(ims, cCoord)
    print('# -------------------try-to-resolve-frames-------------------- #')
    # clds = [Collider(*Vector2.multipleVectors(-4,0, -5,-1, -1,-3, 2,-3, 2,-2, 0,-2))]
    clds2 = [Collider(*Vector2.multipleVectors(-4,-2, 4,-2, 4,-3, -4,-3))]
    # fh = frameHandler(0.01, WIDTH, HEIGHT, sb, clds)
    # fh.resolveFrames(cCoord, 100)
    print('# ------------------test spring------------------------------- #')
    # a=Point(10,Vector2(1,-1),Vector2(0,-1))
    # b=Point(10,Vector2(-1,-1),Vector2(0,-1))
    # s=Spring(a,b,1.5,1)
    # sb2=SoftBody([[a,b]],(s,),g=Vector2(0,0))
    # print(sb2.points)
    # fh2 = frameHandler(0.05,WIDTH,HEIGHT,sb2,clds2,'spring_test')
    # fh2.resolveFrames(cCoord,100)
    print('# -----------------test-collision----------------------------- #')
    dbsb = SoftBody([[Point(1,Vector2(0,0),Vector2(-3,-10))]],[],g=Vector2(0,0)) #debug-use softbody
    dbrf(dbsb,clds2,0.01,cCoord,30)

    ims.write_to_png(os.path.join(os.curdir, f"{FILENAME[:-3]}.png"))
#  少self collision
